using System;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace GameTestFixture
{
    [Serializable]
    public sealed class DebugStateSnapshot
    {
        public string scene;
        public bool door_open;
        public string timestamp;
    }

    public static class DebugStateExporter
    {
        public static string DefaultOutputPath()
        {
            return Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", "..", "results", "unity", "debug_state.json"));
        }

        public static string Export(DoorController door, string outputPath = null)
        {
            if (door == null)
            {
                throw new ArgumentNullException(nameof(door));
            }

            string resolvedPath = outputPath ?? DefaultOutputPath();
            string directory = Path.GetDirectoryName(resolvedPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var snapshot = new DebugStateSnapshot
            {
                scene = SceneManager.GetActiveScene().name,
                door_open = door.IsOpen,
                timestamp = DateTimeOffset.UtcNow.ToString("O"),
            };

            File.WriteAllText(resolvedPath, JsonUtility.ToJson(snapshot, prettyPrint: true));
            return resolvedPath;
        }

        public static string Export(CheckpointDebugState snapshot, string outputPath = null)
        {
            if (snapshot == null)
            {
                throw new ArgumentNullException(nameof(snapshot));
            }

            string resolvedPath = outputPath ?? DefaultOutputPath();
            string directory = Path.GetDirectoryName(resolvedPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            if (string.IsNullOrEmpty(snapshot.scene))
            {
                snapshot.scene = SceneManager.GetActiveScene().name;
            }

            snapshot.timestamp = DateTimeOffset.UtcNow.ToString("O");
            File.WriteAllText(resolvedPath, JsonUtility.ToJson(snapshot, prettyPrint: true));
            return resolvedPath;
        }
    }
}
