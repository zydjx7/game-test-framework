using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class CheckpointRuntimeFixture : IDisposable
    {
        private readonly bool doorNotPersistedBug;
        private readonly List<string> trace = new List<string>();

        private GameObject root;
        private PlayerState player;
        private KeycardPickup keycard;
        private SecurityDoor door;
        private CheckpointMarker checkpoint;
        private DeathRespawn respawn;
        private ExtractionPoint extraction;
        private Camera screenshotCamera;
        private GameObject doorVisual;
        private GameObject playerVisual;
        private bool extractionReached;
        private bool progressionSoftlock;
        private string failureReason = string.Empty;

        public CheckpointRuntimeFixture(bool doorNotPersistedBug)
        {
            this.doorNotPersistedBug = doorNotPersistedBug;
        }

        public CheckpointObservation Reset()
        {
            DestroyRoot();

            root = new GameObject("Gate2RuntimeBridgeFixture");
            player = AddFixtureComponent<PlayerState>("Player");
            keycard = AddFixtureComponent<KeycardPickup>("Keycard");
            checkpoint = AddFixtureComponent<CheckpointMarker>("Checkpoint");
            respawn = AddFixtureComponent<DeathRespawn>("DeathRespawn");
            extraction = AddFixtureComponent<ExtractionPoint>("Extraction");

            var doorObject = new GameObject("SecurityDoor");
            doorObject.transform.SetParent(root.transform);
            doorObject.AddComponent<DoorController>();
            door = doorObject.AddComponent<SecurityDoor>();

            CreateScreenshotRig();

            extractionReached = false;
            progressionSoftlock = false;
            failureReason = string.Empty;
            trace.Clear();
            trace.Add("reset");

            return Observe();
        }

        public CheckpointObservation RunAction(string action)
        {
            EnsureReady();

            switch (action)
            {
                case "move_to_keycard":
                    player.MoveTo("keycard_room");
                    break;
                case "collect_keycard":
                    if (!keycard.Collect(player))
                    {
                        throw new InvalidOperationException("Keycard was already collected.");
                    }
                    break;
                case "move_to_door":
                    player.MoveTo("security_door");
                    break;
                case "open_door":
                    if (!door.TryOpen(player))
                    {
                        throw new InvalidOperationException("Security door did not open.");
                    }
                    break;
                case "move_to_checkpoint":
                    player.MoveTo("checkpoint_room");
                    break;
                case "activate_checkpoint":
                    checkpoint.Capture(player, door, keycard);
                    break;
                case "move_to_hazard":
                    player.MoveTo("hazard_room");
                    break;
                case "die_and_respawn":
                    respawn.KillAndRespawn(player, checkpoint, door, keycard, doorNotPersistedBug);
                    break;
                case "extract":
                    extractionReached = extraction.TryExtract(
                        player,
                        door,
                        out failureReason,
                        out progressionSoftlock);
                    break;
                default:
                    throw new ArgumentException($"Unknown bridge action: {action}", nameof(action));
            }

            trace.Add($"action:{action}");
            return Observe();
        }

        public CheckpointObservation Observe()
        {
            EnsureReady();

            return new CheckpointObservation
            {
                player_zone = player != null ? player.Zone : string.Empty,
                has_keycard = player != null && player.HasKeycard,
                keycard_collected = keycard != null && keycard.IsCollected,
                door_open = door != null && door.IsOpen,
                checkpoint_active = checkpoint != null && checkpoint.IsActivated,
                respawn_count = player != null ? player.RespawnCount : 0,
                extraction_reached = extractionReached,
                progression_softlock = progressionSoftlock,
                bug_door_not_persisted = doorNotPersistedBug,
                failure_reason = failureReason ?? string.Empty,
            };
        }

        public CheckpointDebugState BuildDebugState()
        {
            EnsureReady();

            return CheckpointDebugState.FromFixture(
                player,
                door,
                keycard,
                checkpoint,
                extractionReached,
                progressionSoftlock,
                doorNotPersistedBug,
                failureReason);
        }

        public CheckpointDebugState ExportDebugState()
        {
            var snapshot = BuildDebugState();
            DebugStateExporter.Export(snapshot);
            trace.Add("debug_state");
            return snapshot;
        }

        public string ExportScreenshot()
        {
            EnsureReady();
            UpdateScreenshotVisuals();

            string path = DefaultScreenshotPath();
            string directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var renderTexture = new RenderTexture(320, 180, 24);
            var previousActive = RenderTexture.active;
            var previousTarget = screenshotCamera.targetTexture;
            try
            {
                screenshotCamera.targetTexture = renderTexture;
                RenderTexture.active = renderTexture;
                screenshotCamera.Render();

                var texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
                texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                texture.Apply();
                File.WriteAllBytes(path, texture.EncodeToPNG());
                UnityEngine.Object.Destroy(texture);
            }
            finally
            {
                screenshotCamera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                renderTexture.Release();
                UnityEngine.Object.Destroy(renderTexture);
            }

            trace.Add("screenshot");
            return path;
        }

        public string[] Trace()
        {
            return trace.ToArray();
        }

        public static string DefaultScreenshotPath()
        {
            return Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", "..", "results", "unity", "bridge_screenshot.png"));
        }

        public void Dispose()
        {
            DestroyRoot();
        }

        private T AddFixtureComponent<T>(string name)
            where T : Component
        {
            var gameObject = new GameObject(name);
            gameObject.transform.SetParent(root.transform);
            return gameObject.AddComponent<T>();
        }

        private void EnsureReady()
        {
            if (root == null)
            {
                Reset();
            }
        }

        private void DestroyRoot()
        {
            if (root != null)
            {
                UnityEngine.Object.Destroy(root);
                root = null;
            }
        }

        private void CreateScreenshotRig()
        {
            var cameraObject = new GameObject("BridgeScreenshotCamera");
            cameraObject.transform.SetParent(root.transform);
            cameraObject.transform.position = new Vector3(0f, 4f, -7f);
            cameraObject.transform.rotation = Quaternion.Euler(28f, 0f, 0f);
            screenshotCamera = cameraObject.AddComponent<Camera>();
            screenshotCamera.clearFlags = CameraClearFlags.SolidColor;
            screenshotCamera.backgroundColor = new Color(0.08f, 0.09f, 0.1f);
            screenshotCamera.orthographic = true;
            screenshotCamera.orthographicSize = 4f;

            doorVisual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            doorVisual.name = "DoorVisual";
            doorVisual.transform.SetParent(root.transform);
            doorVisual.transform.position = new Vector3(1.4f, 0.5f, 0f);
            doorVisual.transform.localScale = new Vector3(0.35f, 1f, 0.2f);

            playerVisual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            playerVisual.name = "PlayerVisual";
            playerVisual.transform.SetParent(root.transform);
            playerVisual.transform.localScale = new Vector3(0.35f, 0.35f, 0.35f);

            UpdateScreenshotVisuals();
        }

        private void UpdateScreenshotVisuals()
        {
            if (doorVisual != null)
            {
                SetRendererColor(doorVisual, door != null && door.IsOpen ? Color.green : Color.red);
            }

            if (playerVisual != null)
            {
                playerVisual.transform.position = ZoneToPosition(player != null ? player.Zone : string.Empty);
                SetRendererColor(playerVisual, Color.cyan);
            }
        }

        private static Vector3 ZoneToPosition(string zone)
        {
            switch (zone)
            {
                case "keycard_room":
                    return new Vector3(-2f, 0.35f, 0f);
                case "security_door":
                    return new Vector3(-0.5f, 0.35f, 0f);
                case "checkpoint_room":
                    return new Vector3(0.7f, 0.35f, 0f);
                case "hazard_room":
                    return new Vector3(2f, 0.35f, 0f);
                default:
                    return new Vector3(-3f, 0.35f, 0f);
            }
        }

        private static void SetRendererColor(GameObject target, Color color)
        {
            var renderer = target.GetComponent<Renderer>();
            if (renderer == null)
            {
                return;
            }

            var shader = Shader.Find("Unlit/Color") ?? Shader.Find("Standard");
            renderer.material = new Material(shader)
            {
                color = color,
            };
        }
    }
}
