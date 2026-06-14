using System;
using UnityEngine.SceneManagement;

namespace GameTestFixture
{
    [Serializable]
    public sealed class CheckpointVisualState
    {
        public string scene;
        public bool door_visual_open;
        public string door_visual_color;
        public string door_visual_source;
        public bool bug_door_visual_stuck_closed;

        public static CheckpointVisualState FromDoorVisual(
            DoorVisualController doorVisual,
            bool doorVisualStuckClosedBug)
        {
            return new CheckpointVisualState
            {
                scene = SceneManager.GetActiveScene().name,
                door_visual_open = doorVisual != null && doorVisual.IsVisualOpen,
                door_visual_color = doorVisual != null ? doorVisual.VisualColorName : string.Empty,
                door_visual_source = doorVisual != null ? doorVisual.StateSource : string.Empty,
                bug_door_visual_stuck_closed = doorVisualStuckClosedBug,
            };
        }
    }
}
