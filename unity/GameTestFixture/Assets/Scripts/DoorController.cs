using UnityEngine;

namespace GameTestFixture
{
    public sealed class DoorController : MonoBehaviour
    {
        public bool IsOpen { get; private set; }

        public void Open()
        {
            IsOpen = true;
        }

        public void Close()
        {
            IsOpen = false;
        }
    }
}
