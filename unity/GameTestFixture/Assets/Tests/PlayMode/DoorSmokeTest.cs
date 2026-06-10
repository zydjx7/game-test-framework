using System.Collections;
using System.IO;
using GameTestFixture;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace GameTestFixture.Tests
{
    public sealed class DoorSmokeTest
    {
        [UnityTest]
        public IEnumerator DoorOpenExportsDebugState()
        {
            string outputPath = DebugStateExporter.DefaultOutputPath();
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            var doorObject = new GameObject("DoorUnderTest");
            var door = doorObject.AddComponent<DoorController>();

            Assert.IsFalse(door.IsOpen, "Door should start closed.");

            door.Open();

            Assert.IsTrue(door.IsOpen, "Door should be open after Open().");

            string exportedPath = DebugStateExporter.Export(door);

            Assert.AreEqual(outputPath, exportedPath);
            Assert.IsTrue(File.Exists(exportedPath), "debug_state.json should be exported.");

            string json = File.ReadAllText(exportedPath);
            var snapshot = JsonUtility.FromJson<DebugStateSnapshot>(json);
            Assert.IsNotNull(snapshot, "debug_state.json should parse as a DebugStateSnapshot.");
            Assert.IsTrue(snapshot.door_open, "debug_state.json should report door_open: true.");

            Object.Destroy(doorObject);
            yield return null;
        }
    }
}
