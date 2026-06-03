// Licensed under the MIT License. See LICENSE in the project root for license information.

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenAI.Extensions;
using OpenAI.Models;
using System;
using System.Collections.Generic;
using System.Security.Authentication;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Utilities.Async;
using Utilities.WebRequestRest;
using Utilities.WebSockets;

namespace OpenAI.Realtime
{
    public sealed class RealtimeEndpoint : OpenAIBaseEndpoint
    {
        public RealtimeEndpoint(OpenAIClient client) : base(client) { }

        protected override string Root => "realtime";

        /// <summary>
        /// Creates a new realtime session with the provided <see cref="SessionConfiguration"/> options.
        /// </summary>
        /// <param name="configuration"><see cref="SessionConfiguration"/>.</param>
        /// <param name="cancellationToken">Optional, <see cref="CancellationToken"/>.</param>
        /// <returns><see cref="RealtimeSession"/>.</returns>
        public async Task<RealtimeSession> CreateSessionAsync(SessionConfiguration configuration = null, CancellationToken cancellationToken = default)
        {
            string model = string.IsNullOrWhiteSpace(configuration?.Model) ? Model.GPT_Realtime : configuration!.Model;
            var queryParameters = new Dictionary<string, string>();

            if (client.Settings.Info.IsAzureOpenAI)
            {
                queryParameters["deployment"] = model;
            }
            else
            {
                queryParameters["model"] = model;
            }

            // Build GA API request body by converting beta format to GA format
            var configJson = JObject.Parse(JsonConvert.SerializeObject(configuration, OpenAIClient.JsonSerializationOptions));
            var expiresAfter = configJson.Remove("client_secret") != null
                ? configuration.ClientSecret?.ExpiresAfter ?? new ExpiresAfter(600)
                : new ExpiresAfter(600);

            // Map beta fields to GA "audio" structure
            var audioInput = new JObject();
            var audioOutput = new JObject();

            // input_audio_format -> audio.input.format
            var inputFormat = configJson.Remove("input_audio_format");
            if (inputFormat != null) audioInput["format"] = inputFormat;

            // output_audio_format -> audio.output.format
            var outputFormat = configJson.Remove("output_audio_format");
            if (outputFormat != null) audioOutput["format"] = outputFormat;

            // voice -> audio.output.voice
            var voice = configJson.Remove("voice");
            if (voice != null) audioOutput["voice"] = voice;

            // speed -> audio.output.speed
            var speed = configJson.Remove("speed");
            if (speed != null) audioOutput["speed"] = speed;

            // turn_detection -> audio.input.turn_detection
            var turnDetection = configJson.Remove("turn_detection");
            if (turnDetection != null && turnDetection.Type != JTokenType.Null) audioInput["turn_detection"] = turnDetection;

            // input_audio_noise_reduction -> audio.input.noise_reduction
            var noiseReduction = configJson.Remove("input_audio_noise_reduction");
            if (noiseReduction != null) audioInput["noise_reduction"] = noiseReduction;

            // input_audio_transcription -> audio.input.transcription
            var transcription = configJson.Remove("input_audio_transcription");
            if (transcription != null) audioInput["transcription"] = transcription;

            // modalities -> output_modalities
            var modalities = configJson.Remove("modalities");
            if (modalities != null) configJson["output_modalities"] = modalities;

            // Build audio object
            var audio = new JObject();
            if (audioInput.Count > 0) audio["input"] = audioInput;
            if (audioOutput.Count > 0) audio["output"] = audioOutput;
            if (audio.Count > 0) configJson["audio"] = audio;

            configJson["type"] = "realtime";

            var requestBody = new JObject
            {
                ["expires_after"] = JObject.FromObject(expiresAfter, JsonSerializer.Create(OpenAIClient.JsonSerializationOptions)),
                ["session"] = configJson
            };

            var payload = requestBody.ToString(Formatting.None);
            var createSessionResponse = await Rest.PostAsync(GetUrl("/client_secrets"), payload, new RestParameters(client.DefaultRequestHeaders), cancellationToken);
            createSessionResponse.Validate(EnableDebug);

            // Response: { "client_secret": { "value": "...", "expires_at": ... }, "session": {...} }
            var responseJson = JObject.Parse(createSessionResponse.Body);
            var clientSecretToken = responseJson["client_secret"];

            if (clientSecretToken == null)
            {
                throw new InvalidOperationException("Failed to create a client secret. Response did not contain 'client_secret'.");
            }

            var clientSecret = clientSecretToken.ToObject<ClientSecret>(JsonSerializer.Create(OpenAIClient.JsonSerializationOptions));
            var createSession = configuration;
            createSession.ClientSecret = clientSecret;

            if (string.IsNullOrWhiteSpace(createSession.ClientSecret?.EphemeralApiKey))
            {
                throw new InvalidOperationException("Failed to create a session. Ensure the configuration is valid and the API key is set.");
            }

            var websocket = new WebSocket(GetWebsocketUri(queryParameters: queryParameters), new Dictionary<string, string>
            {
#if !PLATFORM_WEBGL
                { "User-Agent", "OpenAI-DotNet" },
                { "OpenAI-Beta", "realtime=v1" },
                { "Authorization", $"Bearer {createSession.ClientSecret!.EphemeralApiKey}" }
#endif
            }, new List<string>
            {
#if PLATFORM_WEBGL // Web browsers do not support headers. https://github.com/openai/openai-realtime-api-beta/blob/339e9553a757ef1cf8c767272fc750c1e62effbb/lib/api.js#L76-L80
                "realtime",
                $"openai-insecure-api-key.{createSession.ClientSecret!.EphemeralApiKey}",
                "openai-beta.realtime-v1"
#endif
            });
            var session = new RealtimeSession(websocket, EnableDebug);
            var sessionCreatedTcs = new TaskCompletionSource<SessionResponse>();

            try
            {
                session.OnEventReceived += OnEventReceived;
                session.OnError += OnError;
                await session.ConnectAsync(cancellationToken).ConfigureAwait(true);
                var sessionResponse = await sessionCreatedTcs.Task.WithCancellation(cancellationToken).ConfigureAwait(true);
                session.Configuration = sessionResponse.SessionConfiguration;
            }
            finally
            {
                session.OnError -= OnError;
                session.OnEventReceived -= OnEventReceived;
            }

            return session;

            void OnError(Exception e)
                => sessionCreatedTcs.TrySetException(e);

            void OnEventReceived(IRealtimeEvent @event)
            {
                try
                {
                    switch (@event)
                    {
                        case SessionResponse sessionResponse:
                            if (sessionResponse.Type == "session.created")
                            {
                                sessionCreatedTcs.TrySetResult(sessionResponse);
                            }

                            break;
                        case RealtimeEventError realtimeEventError:
                            sessionCreatedTcs.TrySetException(realtimeEventError.Error.Code is "invalid_session_token" or "invalid_api_key"
                                ? new AuthenticationException(realtimeEventError.Error.Message)
                                : new Exception(realtimeEventError.Error.Message));
                            break;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    sessionCreatedTcs.TrySetException(e);
                }
            }
        }
    }
}
