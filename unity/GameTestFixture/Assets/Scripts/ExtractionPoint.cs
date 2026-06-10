using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class ExtractionPoint : MonoBehaviour
    {
        [SerializeField] private string requiredZone = "checkpoint_room";

        public bool TryExtract(
            PlayerState player,
            SecurityDoor door,
            out string failureReason,
            out bool progressionSoftlock)
        {
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }
            if (door == null)
            {
                throw new ArgumentNullException(nameof(door));
            }

            progressionSoftlock = false;

            if (!door.IsOpen)
            {
                progressionSoftlock = true;
                failureReason = "progression_softlock: security door closed after checkpoint respawn";
                return false;
            }

            if (!string.Equals(player.Zone, requiredZone, StringComparison.Ordinal))
            {
                failureReason = $"player is in zone '{player.Zone}', expected '{requiredZone}'";
                return false;
            }

            failureReason = string.Empty;
            return true;
        }
    }
}
