using System;
using System.Collections;
using System.IO;
using GameTestFixture;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace GameTestFixture.Tests
{
    public sealed class RuntimeBridgeHostTest
    {
        private const string HostEnvVar = "GATE2_BRIDGE_HOST";
        private const string ReadyFileEnvVar = "GATE2_BRIDGE_READY_FILE";
        private const string TimeoutEnvVar = "GATE2_BRIDGE_TIMEOUT_SECONDS";
        private const string BugEnvVar = "GATE1_BUG_DOOR_NOT_PERSISTED";

        [UnityTest]
        [Category("Gate2Bridge")]
        public IEnumerator PythonDrivesCheckpointFlowOverRuntimeBridge()
        {
            if (!IsEnabled(HostEnvVar))
            {
                yield break;
            }

            bool doorNotPersistedBug = IsEnabled(BugEnvVar);
            float timeoutSeconds = ReadTimeoutSeconds();
            string readyPath = ReadyFilePath();
            string readyDirectory = Path.GetDirectoryName(readyPath);
            if (!string.IsNullOrEmpty(readyDirectory))
            {
                Directory.CreateDirectory(readyDirectory);
            }

            if (File.Exists(readyPath))
            {
                File.Delete(readyPath);
            }

            var fixture = new CheckpointRuntimeFixture(doorNotPersistedBug);
            RuntimeBridgeServer server = null;
            try
            {
                server = new RuntimeBridgeServer(fixture);
                server.Start();
                fixture.Reset();

                var ready = new BridgeReadyState
                {
                    host = "127.0.0.1",
                    port = server.Port,
                };
                File.WriteAllText(readyPath, JsonUtility.ToJson(ready, prettyPrint: true));

                float deadline = Time.realtimeSinceStartup + timeoutSeconds;
                while (!server.ShutdownRequested && Time.realtimeSinceStartup < deadline)
                {
                    server.Pump();
                    if (!string.IsNullOrEmpty(server.FatalError))
                    {
                        break;
                    }

                    yield return null;
                }

                server.Pump();

                Assert.IsTrue(server.ShutdownRequested, "Python smoke did not request bridge shutdown before timeout.");
                Assert.IsTrue(string.IsNullOrEmpty(server.FatalError), $"Runtime bridge error: {server.FatalError}");
            }
            finally
            {
                if (server != null)
                {
                    server.Stop();
                }

                fixture.Dispose();
            }
        }

        private static bool IsEnabled(string envVar)
        {
            return string.Equals(
                Environment.GetEnvironmentVariable(envVar),
                "1",
                StringComparison.OrdinalIgnoreCase);
        }

        private static float ReadTimeoutSeconds()
        {
            string raw = Environment.GetEnvironmentVariable(TimeoutEnvVar);
            if (float.TryParse(raw, out float value) && value > 0f)
            {
                return value;
            }

            return 60f;
        }

        private static string ReadyFilePath()
        {
            string configured = Environment.GetEnvironmentVariable(ReadyFileEnvVar);
            if (!string.IsNullOrEmpty(configured))
            {
                return configured;
            }

            return Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", "..", "results", "unity", "bridge_ready.json"));
        }
    }
}
