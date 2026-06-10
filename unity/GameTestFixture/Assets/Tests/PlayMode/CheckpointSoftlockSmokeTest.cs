using System;
using System.Collections;
using System.IO;
using GameTestFixture;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace GameTestFixture.Tests
{
    public sealed class CheckpointSoftlockSmokeTest
    {
        private const string BugEnvVar = "GATE1_BUG_DOOR_NOT_PERSISTED";

        [UnityTest]
        public IEnumerator FixedScriptReachesExtractionAfterCheckpointRespawn()
        {
            string outputPath = DebugStateExporter.DefaultOutputPath();
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            bool doorNotPersistedBug = string.Equals(
                Environment.GetEnvironmentVariable(BugEnvVar),
                "1",
                StringComparison.OrdinalIgnoreCase);

            var root = new GameObject("Gate1CheckpointSoftlockFixture");
            var player = AddFixtureComponent<PlayerState>(root, "Player");
            var keycard = AddFixtureComponent<KeycardPickup>(root, "Keycard");
            var checkpoint = AddFixtureComponent<CheckpointMarker>(root, "Checkpoint");
            var respawn = AddFixtureComponent<DeathRespawn>(root, "DeathRespawn");
            var extraction = AddFixtureComponent<ExtractionPoint>(root, "Extraction");

            var doorObject = new GameObject("SecurityDoor");
            doorObject.transform.SetParent(root.transform);
            doorObject.AddComponent<DoorController>();
            var door = doorObject.AddComponent<SecurityDoor>();

            try
            {
                player.MoveTo("keycard_room");
                Assert.IsTrue(keycard.Collect(player), "Player should collect the keycard once.");
                Assert.IsTrue(player.HasKeycard, "Keycard pickup should grant inventory.");

                player.MoveTo("security_door");
                Assert.IsTrue(door.TryOpen(player), "Security door should open with the keycard.");
                Assert.IsTrue(door.IsOpen, "Door should be open after keycard use.");
                Assert.IsFalse(player.HasKeycard, "Door use consumes the keycard in this fixture.");

                player.MoveTo("checkpoint_room");
                checkpoint.Capture(player, door, keycard);
                Assert.IsTrue(checkpoint.IsActivated, "Checkpoint should capture post-door state.");

                player.MoveTo("hazard_room");
                respawn.KillAndRespawn(player, checkpoint, door, keycard, doorNotPersistedBug);
                Assert.AreEqual(1, player.RespawnCount, "Death should respawn exactly once.");
                Assert.AreEqual("checkpoint_room", player.Zone, "Respawn should return to checkpoint.");

                bool extractionReached = extraction.TryExtract(
                    player,
                    door,
                    out string failureReason,
                    out bool progressionSoftlock);

                var snapshot = CheckpointDebugState.FromFixture(
                    player,
                    door,
                    keycard,
                    checkpoint,
                    extractionReached,
                    progressionSoftlock,
                    doorNotPersistedBug,
                    failureReason);
                string exportedPath = DebugStateExporter.Export(snapshot);

                Assert.AreEqual(outputPath, exportedPath);
                Assert.IsTrue(File.Exists(exportedPath), "debug_state.json should be exported.");

                string json = File.ReadAllText(exportedPath);
                var exported = JsonUtility.FromJson<CheckpointDebugState>(json);
                Assert.IsNotNull(exported, "debug_state.json should parse as checkpoint state.");
                Assert.IsTrue(exported.checkpoint_active, "debug_state should record checkpoint_active.");
                Assert.AreEqual(1, exported.respawn_count, "debug_state should record one respawn.");

                Assert.IsTrue(
                    extractionReached,
                    "Fixed script should reach extraction after checkpoint respawn. " +
                    $"bug={doorNotPersistedBug}; reason={failureReason}");
                Assert.IsFalse(progressionSoftlock, "Normal fixture should not report progression_softlock.");
                Assert.IsTrue(exported.door_open, "debug_state should record door_open: true.");
                Assert.IsTrue(exported.extraction_reached, "debug_state should record extraction_reached: true.");
                Assert.IsFalse(exported.progression_softlock, "debug_state should record progression_softlock: false.");
                Assert.IsFalse(exported.has_keycard, "Consumed keycard should remain consumed after respawn.");
                Assert.IsTrue(exported.keycard_collected, "Collected keycard should stay collected after respawn.");
            }
            finally
            {
                UnityEngine.Object.Destroy(root);
            }

            yield return null;
        }

        private static T AddFixtureComponent<T>(GameObject root, string name)
            where T : Component
        {
            var gameObject = new GameObject(name);
            gameObject.transform.SetParent(root.transform);
            return gameObject.AddComponent<T>();
        }
    }
}
