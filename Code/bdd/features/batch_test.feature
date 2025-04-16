Feature: Weapon System in the Game

  Scenario: Equipping and Firing a Weapon
    Given the game is started
    When player equips a weapon
    Then the crosshair should be visible
    When player fires the weapon
    Then the ammo count should decrease

  Scenario: Switching Weapons
    Given the game is started
    When player equips a weapon
    Then the crosshair should be visible
    When player switches to a different weapon
    Then the crosshair should be visible
    When player fires the weapon
    Then the ammo count should decrease

  Scenario: Reloading a Weapon
    Given the game is started
    When player equips a weapon
    Then the crosshair should be visible
    When player fires the weapon
    Then the ammo count should decrease
    When player reloads the weapon
    Then the ammo count should match the expected value