using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class CheckpointMarker : MonoBehaviour
    {
        public bool IsActivated { get; private set; }
        public string SavedZone { get; private set; }
        public bool SavedPlayerHasKeycard { get; private set; }
        public bool SavedKeycardCollected { get; private set; }
        public bool SavedDoorOpen { get; private set; }

        public void Capture(PlayerState player, SecurityDoor door, KeycardPickup keycard)
        {
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }
            if (door == null)
            {
                throw new ArgumentNullException(nameof(door));
            }
            if (keycard == null)
            {
                throw new ArgumentNullException(nameof(keycard));
            }

            IsActivated = true;
            SavedZone = player.Zone;
            SavedPlayerHasKeycard = player.HasKeycard;
            SavedKeycardCollected = keycard.IsCollected;
            SavedDoorOpen = door.IsOpen;
        }

        public void Restore(
            PlayerState player,
            SecurityDoor door,
            KeycardPickup keycard,
            bool doorNotPersistedBug)
        {
            if (!IsActivated)
            {
                throw new InvalidOperationException("Checkpoint must be activated before restore.");
            }
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }
            if (door == null)
            {
                throw new ArgumentNullException(nameof(door));
            }
            if (keycard == null)
            {
                throw new ArgumentNullException(nameof(keycard));
            }

            player.MoveTo(SavedZone);
            player.RestoreInventory(SavedPlayerHasKeycard);
            keycard.RestoreCollected(SavedKeycardCollected);
            door.RestoreOpenState(doorNotPersistedBug ? false : SavedDoorOpen);
        }
    }
}
