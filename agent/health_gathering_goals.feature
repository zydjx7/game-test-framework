# ViZDoom health_gathering goals. Run with VizDoomEnv(scenario="health_gathering").

Scenario: Waiting reduces health
  Goal: Wait and confirm that health decreases over time.
  Available actions: wait_and_check_health
  Success: health_before - health_after >= 1
