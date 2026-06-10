using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class RuntimeBridgeServer : IDisposable
    {
        private sealed class PendingRequest
        {
            public BridgeRequest request;
            public ManualResetEvent done;
            public string responseJson;
        }

        private readonly CheckpointRuntimeFixture fixture;
        private readonly Queue<PendingRequest> pendingRequests = new Queue<PendingRequest>();
        private readonly object syncRoot = new object();

        private TcpListener listener;
        private Thread listenerThread;
        private volatile bool stopRequested;
        private volatile bool shutdownRequested;
        private volatile string fatalError;

        public RuntimeBridgeServer(CheckpointRuntimeFixture fixture)
        {
            this.fixture = fixture ?? throw new ArgumentNullException(nameof(fixture));
        }

        public int Port { get; private set; }

        public bool ShutdownRequested
        {
            get { return shutdownRequested; }
        }

        public string FatalError
        {
            get { return fatalError; }
        }

        public void Start()
        {
            listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            Port = ((IPEndPoint)listener.LocalEndpoint).Port;

            listenerThread = new Thread(ListenLoop)
            {
                IsBackground = true,
                Name = "GameTestFixtureRuntimeBridge",
            };
            listenerThread.Start();
        }

        public void Pump()
        {
            while (true)
            {
                PendingRequest pending;
                lock (syncRoot)
                {
                    if (pendingRequests.Count == 0)
                    {
                        return;
                    }

                    pending = pendingRequests.Dequeue();
                }

                BridgeResponse response;
                try
                {
                    response = Handle(pending.request);
                }
                catch (Exception exception)
                {
                    response = BridgeResponse.Failure(
                        pending.request != null ? pending.request.command : "unknown",
                        exception.Message);
                }

                pending.responseJson = JsonUtility.ToJson(response);
                pending.done.Set();
            }
        }

        public void Stop()
        {
            stopRequested = true;

            if (listener != null)
            {
                listener.Stop();
            }

            if (listenerThread != null && listenerThread.IsAlive)
            {
                listenerThread.Join(1000);
            }
        }

        public void Dispose()
        {
            Stop();
        }

        private BridgeResponse Handle(BridgeRequest request)
        {
            if (request == null || string.IsNullOrEmpty(request.command))
            {
                return BridgeResponse.Failure("unknown", "Bridge request missing command.");
            }

            string command = request.command;
            BridgeResponse response;

            switch (command)
            {
                case "reset":
                    response = BridgeResponse.Success(command, fixture.Reset());
                    break;
                case "action":
                    response = BridgeResponse.Success(command, fixture.RunAction(request.action));
                    break;
                case "observe":
                    response = BridgeResponse.Success(command, fixture.Observe());
                    break;
                case "debug_state":
                    response = BridgeResponse.Success(command, fixture.Observe());
                    response.debug_state = fixture.ExportDebugState();
                    break;
                case "screenshot":
                    response = BridgeResponse.Success(command, fixture.Observe());
                    response.screenshot_path = fixture.ExportScreenshot();
                    break;
                case "trace":
                    response = BridgeResponse.Success(command, fixture.Observe());
                    response.trace = fixture.Trace();
                    break;
                case "shutdown":
                    shutdownRequested = true;
                    response = BridgeResponse.Success(command, fixture.Observe());
                    response.trace = fixture.Trace();
                    break;
                default:
                    response = BridgeResponse.Failure(command, $"Unknown bridge command: {command}");
                    break;
            }

            return response;
        }

        private void ListenLoop()
        {
            try
            {
                using (TcpClient client = listener.AcceptTcpClient())
                using (NetworkStream stream = client.GetStream())
                using (var reader = new StreamReader(stream))
                using (var writer = new StreamWriter(stream))
                {
                    writer.NewLine = "\n";
                    writer.AutoFlush = true;

                    while (!stopRequested)
                    {
                        string line = reader.ReadLine();
                        if (line == null)
                        {
                            break;
                        }

                        BridgeRequest request;
                        try
                        {
                            request = JsonUtility.FromJson<BridgeRequest>(line);
                        }
                        catch (Exception exception)
                        {
                            writer.WriteLine(JsonUtility.ToJson(BridgeResponse.Failure("parse", exception.Message)));
                            continue;
                        }

                        var pending = new PendingRequest
                        {
                            request = request,
                            done = new ManualResetEvent(false),
                        };

                        lock (syncRoot)
                        {
                            pendingRequests.Enqueue(pending);
                        }

                        if (!pending.done.WaitOne(TimeSpan.FromSeconds(30)))
                        {
                            writer.WriteLine(JsonUtility.ToJson(BridgeResponse.Failure("timeout", "Unity bridge request timed out.")));
                            fatalError = "Unity bridge request timed out.";
                            break;
                        }

                        writer.WriteLine(pending.responseJson);

                        if (request != null && string.Equals(request.command, "shutdown", StringComparison.Ordinal))
                        {
                            break;
                        }
                    }
                }
            }
            catch (SocketException exception)
            {
                if (!stopRequested)
                {
                    fatalError = exception.Message;
                }
            }
            catch (Exception exception)
            {
                fatalError = exception.Message;
            }
        }
    }
}
