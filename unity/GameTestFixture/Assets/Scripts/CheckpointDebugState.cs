using System;
using UnityEngine.SceneManagement;

namespace GameTestFixture
{
    [Serializable]
    public sealed class CheckpointDebugState
    {
        public string scene;
        public bool door_open;
        public string player_zone;
        public bool has_keycard;
        public bool keycard_collected;
        public bool checkpoint_active;
        public int respawn_count;
        public bool extraction_reached;
        public bool progression_softlock;
        public bool bug_door_not_persisted;
        public string failure_reason;
        public string timestamp;

        public static CheckpointDebugState FromFixture(
            PlayerState player,
            SecurityDoor door,
            KeycardPickup keycard,
            CheckpointMarker checkpoint,
            bool extractionReached,
            bool progressionSoftlock,
            bool doorNotPersistedBug,
            string failureReason)
        {
            return new CheckpointDebugState
            {
                scene = SceneManager.GetActiveScene().name,
                door_open = door != null && door.IsOpen,
                player_zone = player != null ? player.Zone : string.Empty,
                has_keycard = player != null && player.HasKeycard,
                keycard_collected = keycard != null && keycard.IsCollected,
                checkpoint_active = checkpoint != null && checkpoint.IsActivated,
                respawn_count = player != null ? player.RespawnCount : 0,
                extraction_reached = extractionReached,
                progression_softlock = progressionSoftlock,
                bug_door_not_persisted = doorNotPersistedBug,
                failure_reason = failureReason ?? string.Empty,
            };
        }
    }
}
