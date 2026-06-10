using System;

namespace GameTestFixture
{
    [Serializable]
    public sealed class CheckpointObservation
    {
        public string player_zone;
        public bool has_keycard;
        public bool keycard_collected;
        public bool door_open;
        public bool checkpoint_active;
        public int respawn_count;
        public bool extraction_reached;
        public bool progression_softlock;
        public bool bug_door_not_persisted;
        public string failure_reason;
    }
}
