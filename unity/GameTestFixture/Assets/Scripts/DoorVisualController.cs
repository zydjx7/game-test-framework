using UnityEngine;

namespace GameTestFixture
{
    public sealed class DoorVisualController : MonoBehaviour
    {
        private static readonly Color OpenColor = Color.green;
        private static readonly Color ClosedColor = Color.red;

        [SerializeField] private Renderer targetRenderer;
        private Material visualMaterial;

        public bool IsVisualOpen { get; private set; }

        public string VisualColorName { get; private set; } = "closed_red";

        public string StateSource
        {
            get { return "DoorVisualController.Renderer.material.color"; }
        }

        public void SyncFromDoor(SecurityDoor door, bool forceClosedVisual)
        {
            bool logicOpen = door != null && door.IsOpen;
            SetVisualOpen(logicOpen && !forceClosedVisual);
        }

        private void Awake()
        {
            EnsureRenderer();
            SetVisualOpen(false);
        }

        private void SetVisualOpen(bool isOpen)
        {
            IsVisualOpen = isOpen;
            VisualColorName = isOpen ? "open_green" : "closed_red";

            var renderer = EnsureRenderer();
            if (renderer == null)
            {
                return;
            }

            var material = EnsureMaterial(renderer);
            material.color = isOpen ? OpenColor : ClosedColor;
        }

        private Material EnsureMaterial(Renderer renderer)
        {
            if (visualMaterial != null)
            {
                return visualMaterial;
            }

            var shader = Shader.Find("Unlit/Color") ?? Shader.Find("Standard");
            visualMaterial = new Material(shader)
            {
                name = "DoorVisualRuntimeMaterial",
            };
            renderer.material = visualMaterial;
            return visualMaterial;
        }

        private Renderer EnsureRenderer()
        {
            if (targetRenderer == null)
            {
                targetRenderer = GetComponent<Renderer>();
            }

            return targetRenderer;
        }

        private void OnDestroy()
        {
            if (visualMaterial == null)
            {
                return;
            }

            if (Application.isPlaying)
            {
                Destroy(visualMaterial);
            }
            else
            {
                DestroyImmediate(visualMaterial);
            }

            visualMaterial = null;
        }
    }
}
