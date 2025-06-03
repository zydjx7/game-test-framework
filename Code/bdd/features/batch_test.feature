Feature: Crosshair state verification during death and reload
    Verify crosshair displays correctly during player death and weapon reload scenarios

Scenario: Crosshair should indicate death state when player dies
    Given the game is started
    When player equips a primary weapon
    When player dies
    Then the player view should indicate death

Scenario: Crosshair should show reloading state during weapon reload
    Given the game is started
    When player equips a primary weapon
    When player fires the weapon
    When player reloads the weapon
    Then the crosshair should indicate reloading

Scenario: Crosshair should return to primary weapon state after reload
    Given the game is started
    When player equips a primary weapon
    When player fires the weapon
    When player reloads the weapon
    Then the crosshair should be for a primary weapon

Scenario: Crosshair should remain visible when switching weapons after death
    Given the game is started
    When player equips a primary weapon
    When player dies
    When player switches to knife
    Then the crosshair should be visible

Scenario: Crosshair should show correct state when reloading after weapon switch
    Given the game is started
    When player equips a primary weapon
    When player switches to secondary weapon
    When player reloads the weapon
    Then the crosshair should be for a secondary weapon