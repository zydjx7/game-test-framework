using System;
using UnityEngine;

namespace GameTestFixture
{
    public sealed class PlayerState : MonoBehaviour
    {
        public string Zone { get; private set; } = "start";
        public bool HasKeycard { get; private set; }
        public int RespawnCount { get; private set; }

        public void MoveTo(string zone)
        {
            if (string.IsNullOrWhiteSpace(zone))
            {
                throw new ArgumentException("Zone must be non-empty.", nameof(zone));
            }

            Zone = zone;
        }

        public void GrantKeycard()
        {
            HasKeycard = true;
        }

        public bool ConsumeKeycard()
        {
            if (!HasKeycard)
            {
                return false;
            }

            HasKeycard = false;
            return true;
        }

        public void RestoreInventory(bool hasKeycard)
        {
            HasKeycard = hasKeycard;
        }

        public void RecordRespawn()
        {
            RespawnCount += 1;
        }
    }
}
