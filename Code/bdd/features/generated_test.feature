Feature: Ammo management when player fires a weapon

  Scenario: Ammo decreases when player fires a weapon
    Given the game is started
    When player equips a weapon
    When player fires the weapon
    Then the ammo count should decrease