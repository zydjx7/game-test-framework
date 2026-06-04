# ToyFPS goals — multi-metric (ammo decreases, score increases, health increases).
# Proves the generalized schema and the SAME agent layer handle non-ammo metrics.

Scenario: Firing consumes ammo
  Goal: Fire and confirm it consumes ammo.
  Available actions: fire_and_check_ammo, heal_and_check_health
  Success: ammo_before - ammo_after >= 1

Scenario: Firing increases score
  Goal: Fire and confirm the score goes up.
  Available actions: fire_and_check_score, heal_and_check_health
  Success: score_after - score_before >= 1

Scenario: Healing restores health
  Goal: Heal and confirm health increases.
  Available actions: fire_and_check_ammo, heal_and_check_health
  Success: health_after - health_before >= 5
