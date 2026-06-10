using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class KeycardPickup : MonoBehaviour
    {
        public bool IsCollected { get; private set; }

        public bool Collect(PlayerState player)
        {
            if (player == null)
            {
                throw new ArgumentNullException(nameof(player));
            }

            if (IsCollected)
            {
                return false;
            }

            IsCollected = true;
            player.GrantKeycard();
            return true;
        }

        public void RestoreCollected(bool isCollected)
        {
            IsCollected = isCollected;
        }
    }
}
