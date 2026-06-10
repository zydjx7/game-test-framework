using System;

namespace GameTestFixture
{
    [Serializable]
    public sealed class BridgeRequest
    {
        public string command;
        public string action;
    }

    [Serializable]
    public sealed class BridgeResponse
    {
        public bool ok;
        public string command;
        public string error;
        public CheckpointObservation observation;
        public CheckpointDebugState debug_state;
        public string screenshot_path;
        public string[] trace;

        public static BridgeResponse Success(string command, CheckpointObservation observation)
        {
            return new BridgeResponse
            {
                ok = true,
                command = command,
                error = string.Empty,
                observation = observation,
            };
        }

        public static BridgeResponse Failure(string command, string error)
        {
            return new BridgeResponse
            {
                ok = false,
                command = command,
                error = error ?? string.Empty,
            };
        }
    }

    [Serializable]
    public sealed class BridgeReadyState
    {
        public string host;
        public int port;
    }
}
