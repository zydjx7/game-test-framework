Feature: Crosshair Behavior Verification
Scenario: Verify crosshair changes when aiming at teammates and enemies
    Given the game is started
    When player equips a primary weapon
    Then the crosshair should be for a primary weapon
    When player aims at an enemy
    Then the crosshair should indicate aiming at an enemy
    When player aims at a teammate
    Then the crosshair should indicate aiming at a teammate