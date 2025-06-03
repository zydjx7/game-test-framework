Feature: 使用LLM进行弹药分析测试
  为了演示LLM在游戏测试中的优势
  作为测试人员
  我希望LLM能够帮助我分析游戏状态并选择合适的测试资源

  Scenario: 使用LLM验证开火后弹药减少
    Given the game is started
    And the player has 20 ammo
    When player fires the weapon
    Then the ammo count should be verified by LLM

  Scenario: 使用LLM验证特定弹药数量
    Given the game is started 
    And the player has 15 ammo
    Then the ammo count should be verified by LLM

  Scenario: 多次开火后使用LLM验证
    Given the game is started
    And the player has 10 ammo
    When player fires the weapon
    And player fires the weapon
    Then the ammo count should be verified by LLM 