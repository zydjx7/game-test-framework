# Unity GameTestFixture goals.

Scenario: gameplay_checkpoint_no_softlock
  Goal: Reach extraction after checkpoint respawn without a progression softlock.
  Available actions: collect_keycard, open_security_door, activate_checkpoint, die_and_respawn, attempt_extraction
  Success: extraction_reached_after - extraction_reached_before >= 1 and progression_softlock_after == 0
