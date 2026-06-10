using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class DeathRespawn : MonoBehaviour
    {
        public void KillAndRespawn(
            PlayerState player,
            CheckpointMarker checkpoint,
            SecurityDoor door,
            KeycardPickup keycard,
            bool doorNotPersistedBug)
        {
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }
            if (checkpoint == null)
            {
                throw new ArgumentNullException(nameof(checkpoint));
            }

            checkpoint.Restore(player, door, keycard, doorNotPersistedBug);
            player.RecordRespawn();
        }
    }
}
