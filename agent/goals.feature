# Phase 2 v1 goals (goal-level Gherkin). Each gives the agent BOTH actions so
# it must CHOOSE correctly. Success is judged on the loop's cumulative result.

Scenario: Firing consumes ammo
  Goal: Fire the weapon and confirm that firing consumes ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1

Scenario: Idle does not consume ammo
  Goal: Confirm that staying idle and not firing leaves ammo unchanged.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after == 0 and steps >= 1

Scenario: Repeated firing reduces ammo by three
  Goal: Fire repeatedly until ammo has dropped by at least three.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 3
