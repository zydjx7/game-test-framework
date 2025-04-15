Feature: 弹药系统测试
  
  Scenario: 检查开火后弹药数量减少
    Given the game is started
    When player equips a weapon
    And player fires the weapon
    Then the ammo count should decrease
    
  Scenario: 检查弹药数量显示与内部状态同步
    Given the game is started
    When player equips a weapon
    Then the ammo count should match the expected value 