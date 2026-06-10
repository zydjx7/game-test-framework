using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class SecurityDoor : MonoBehaviour
    {
        [SerializeField] private DoorController door;
        [SerializeField] private bool consumeKeycard = true;

        public bool IsOpen => Door.IsOpen;

        private DoorController Door
        {
            get
            {
                if (door == null)
                {
                    door = GetComponent<DoorController>();
                }

                if (door == null)
                {
                    door = gameObject.AddComponent<DoorController>();
                }

                return door;
            }
        }

        public bool TryOpen(PlayerState player)
        {
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }

            if (IsOpen)
            {
                return true;
            }

            if (!player.HasKeycard)
            {
                return false;
            }

            if (consumeKeycard)
            {
                player.ConsumeKeycard();
            }

            Door.Open();
            return true;
        }

        public void RestoreOpenState(bool isOpen)
        {
            if (isOpen)
            {
                Door.Open();
                return;
            }

            Door.Close();
        }
    }
}
