Feature: Ammo HUD
  Scenario: Spawn with 50 bullets
    Given the game is started
    When player equips a weapon
    Then the crosshair should be visible
    And the ammo count should be 50