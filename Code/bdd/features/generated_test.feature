Feature: Weapon Firing with Full Ammo

Scenario: Player fires weapon with full ammo
    Given the game is started
    When player equips a weapon
    Then the crosshair should be visible
    When player fires the weapon
    Then the ammo count should decrease
    Then the ammo count should match the expected value