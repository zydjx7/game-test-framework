// DecisionMaking.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include "DecisionMakingLayer.h"
#include "Utils.h"
#include <math.h>
#include <random>

using namespace std;

vector<float> GetRandomProbabilities()
{

	std::default_random_engine generator;
	std::discrete_distribution<> distrib({ 15, 5, 15, 10, 15, 5, 10, 5, 10 });
	vector<float> v;
	v.resize(DecisionMakingActions::COUNT);

	for (int i = 0; i < 10000; i++)
	{
		int number = distrib(generator);
		++v[number];
	}

	for (int i = 0; i < v.size(); i++)
	{
		v[i] = v[i] / 10000;
	}

	return v;
}

DecisionMakingActions RandomDecisionMaking::GetCurrentAction()
{
	double rndNumber = rand() / (double)RAND_MAX;
	double offset = 0.0f;
	DecisionMakingActions pick = DecisionMakingActions::NO_ACTION;

	vector<float> randomProbabilities = GetRandomProbabilities();

	for (int i = 0; i < DecisionMakingActions::COUNT; i++)
	{
		offset += randomProbabilities[i];
		if (rndNumber < offset)
		{
			pick = (DecisionMakingActions)i;
			break;
		}
	}

	return pick;
}

RandomDecisionMaking::RandomDecisionMaking()
{
	//m_InputSendingLayer = InputSendingLayerImpl();
	//Strategy();
}


Actions DecisionMakingLayer::Execute(ComputerVisionLayer::VisionOutput& input)
{
	return m_Strategy->Execute(input);
}

void DecisionMakingLayer::SetStrategy(int _type)
{
	switch (_type)
	{
	case Random:
	{
		m_Strategy = new RandomDecisionMaking();
		break;
	}
	case AIBehavior:
	{
		// TODO
		// m_Strategy = new TreeBehaviorDecisionMaking();
		break;
	}
	default:
		break;
	}
}

void BuildStructure(DecisionMakingActions _actionToDo, Actions &_actions, float _fAngle, float _fPower = 0.0f)
{
	_actions.fStickAngle = _fAngle; // to move forward
	_actions.fPower = _fPower;
	switch (_actionToDo)
	{
	case DRIBBLE:
		_actions.bDribble = true;
		break;
	case SHOOT:
		_actions.bShoot = true;
		_actions.fPower = rand() / (float)RAND_MAX;
		break;
	case SHOOT_CHIP:
		_actions.bShootChip = true;
		_actions.fPower = rand() / (float)RAND_MAX;
		break;
	case SHOOT_DRIVEN:
		_actions.bShootDriven = true;
		_actions.fPower = rand() / (float)RAND_MAX;
		break;
	case PASS:
		_actions.bPass = true;
		break;
	case TACKLE_SLIDE:
		_actions.bTackleSlide = true;
		break;
	case TACKLE_STAND:
		_actions.bTackleStand = true;
		break;
	case SPRINT:
		_actions.bSprint = true;
		break;
	case NO_ACTION:
		_actions.bNoAction = true;
		break;
	case COUNT:
		break;
	default:
		break;
	}
}

Actions RandomDecisionMaking::Execute(ComputerVisionLayer::VisionOutput& input)
{
	DecisionMakingActions eCurrAction = GetCurrentAction();
	Actions actionsToDo;
	BuildStructure(eCurrAction, actionsToDo, 90.0f);

	// TODO: use the fucking input !!!
	if (input.m_isGoalVisible)
	{
		actionsToDo.reset();
		actionsToDo.fStickAngle = 90.0f;
		actionsToDo.bShoot = true;
		actionsToDo.fPower = 1.0f;
	}

	return actionsToDo;
}

Actions UtilityAI::Execute(ComputerVisionLayer::VisionOutput& _input)
{
	Actions actionsToDo;

	ProcessBallHolder(_input);

	float fprobability = std::numeric_limits<float>::min();

	actionsToDo.reset();

	float fProbabilityToPass = PassBehavior(_input);
	float fProbabilityToShoot = ShootBehavior(_input);
	float fProbabilityToDribble = DribbleBehavior(_input);
	float fProbabilityToTackle = TackleBehavior(_input);

	vector<std::pair<DecisionMakingActions, float>> probabilities;
	probabilities.push_back(std::make_pair(PASS, fProbabilityToPass));
	probabilities.push_back(std::make_pair(SHOOT, fProbabilityToPass));
	probabilities.push_back(std::make_pair(DRIBBLE, fProbabilityToPass));
	probabilities.push_back(std::make_pair(TACKLE_STAND, fProbabilityToPass));

	DecisionMakingActions typeActionToExecute = NO_ACTION;
	for (int i = 0; i < probabilities.size(); i++)
	{
		if (probabilities[i].first > fprobability)
		{
			fprobability = probabilities[i].second; 
			typeActionToExecute = probabilities[i].first;
		}
	}

	float fAngle = 0.0f;
	switch (typeActionToExecute)
	{
	case DecisionMakingActions::PASS:
		fAngle = (atan2(m_PlayerToPass[1], m_PlayerToPass[0]) / M_PI) * 180.0f;
		BuildStructure(DecisionMakingActions::PASS, actionsToDo, fAngle, m_fPowerToShoot);
		break;
	case DecisionMakingActions::SHOOT:
		fAngle = (atan2(m_PositionToShoot[1], m_PositionToShoot[0]) / M_PI) * 180.0f;
		BuildStructure(DecisionMakingActions::TACKLE_STAND, actionsToDo, fAngle, m_fPowerToShoot);
		break;
	case DecisionMakingActions::TACKLE_STAND:
		fAngle = (atan2(m_PlayerToTackle[1], m_PlayerToTackle[0]) / M_PI) * 180.0f;
		BuildStructure(DecisionMakingActions::TACKLE_STAND, actionsToDo, fAngle, m_fPowerToShoot);
		break;
	case DecisionMakingActions::DRIBBLE:
		fAngle = (atan2(m_PositionToDribble[1], m_PositionToDribble[0]) / M_PI) * 180.0f;
		BuildStructure(DecisionMakingActions::DRIBBLE, actionsToDo, fAngle, m_fPowerToShoot);
		break;
	case DecisionMakingActions::NO_ACTION:
		break;
	default:
		break;
	}

	return actionsToDo;
}


float AngleBetween(const cv::Point2f &v1, const cv::Point2f &v2)
{
	float len1 = sqrtf(v1.x * v1.x + v1.y * v1.y);
	float len2 = sqrtf(v2.x * v2.x + v2.y * v2.y);

	float dot = v1.x * v2.x + v1.y * v2.y;

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return (float)M_PI;
	else
		return acos(a); // 0..PI
}

void UtilityAI::ProcessBallHolder(ComputerVisionLayer::VisionOutput& _computerVisionOutput)
{
	std::vector<ComputerVisionLayer::VisionOutput::BallInfo> ballCoordinates;
	std::vector<ComputerVisionLayer::VisionOutput::PlayerInfo> homePlayers;
	std::vector<ComputerVisionLayer::VisionOutput::PlayerInfo> awayPlayers;

	if (_computerVisionOutput.m_isRealCoordinates)
	{
		ballCoordinates = _computerVisionOutput.m_ballCoordinates_real;
		homePlayers = _computerVisionOutput.m_homePlayers_real;
		awayPlayers = _computerVisionOutput.m_awayPlayers_real;
	}
	else
	{
		ballCoordinates = _computerVisionOutput.m_ballCoordinates_img;
		homePlayers = _computerVisionOutput.m_homePlayers_img;
		awayPlayers = _computerVisionOutput.m_awayPlayers_img;
	}

	// Distance between ball and the closest home team player and away team player
	double distBallHomeTeam = std::numeric_limits<double>::max();
	double distBallAwayTeam = std::numeric_limits<double>::max();

	// The positions of the both closest players from the ball.
	cv::Point2i closestPlayerHome;
	cv::Point2i closestPlayerAway;

	for (const cv::Point2i& ballPos : ballCoordinates)
	{
		for (const cv::Point2i& playerHomePos : homePlayers)
		{
			double res = cv::norm(ballPos - playerHomePos);
			if (res < distBallHomeTeam)
			{
				distBallHomeTeam = res;
				closestPlayerHome = playerHomePos;
			}
		}

		for (const cv::Point2i& playerAwayPos : awayPlayers)
		{
			double res = cv::norm(ballPos - playerAwayPos);
			if (res < distBallAwayTeam)
			{
				distBallAwayTeam = res;
				closestPlayerAway = playerAwayPos;
			}
		}
	}

	m_bIsOffensiveTeam = distBallHomeTeam < distBallAwayTeam;
	m_BallPossessorCoordinates = m_bIsOffensiveTeam ? closestPlayerHome : closestPlayerAway; // Save the position of ball possessor
	m_BallDefensiveHolder = m_bIsOffensiveTeam ? closestPlayerAway : closestPlayerHome;
}

/*void UtilityAI::StrategyImpl()
{

}*/

#define MIN_DISTANCE_TO_PASS 0.0f
#define MAX_DISTANCE_TO_PASS 500.0f

#define MIN_DISTANCE_FROM_AWAY 10.0f
#define MAX_DISTANCE_FROM_AWAY 1000.0f

#define MIN_PROBABILITY 0.1f
#define MAX_PROBABILITY 1.0f

float lerp(float _t, float _start, float _end) {
	return (1 - _t) * _start + _t * _end;
}

float InterpolateProbabilitiesBetweenIntervals(float _value, float _minFrom, float _maxFrom, float _minTo, float _maxTo)
{
	float procent = (_maxFrom + _minFrom) / _value;
	return lerp(procent, _minTo, _maxTo);
}

float UtilityAI::PassBehavior(ComputerVisionLayer::VisionOutput& _computerVisionOutput)
{
	if (m_bIsOffensiveTeam)
	{
		std::vector<ComputerVisionLayer::VisionOutput::PlayerInfo> awayPlayers;
		std::vector<ComputerVisionLayer::VisionOutput::PlayerInfo> homePlayers;

		if (_computerVisionOutput.m_isRealCoordinates)
		{
			awayPlayers = _computerVisionOutput.m_awayPlayers_real;
			homePlayers = _computerVisionOutput.m_homePlayers_real;
		}
		else
		{
			awayPlayers = _computerVisionOutput.m_awayPlayers_img;
			homePlayers = _computerVisionOutput.m_homePlayers_img;
		}
		
		float fProbability = std::numeric_limits<double>::min();

		for (const cv::Point2i& playerHomePos : homePlayers)
		{
			float fDistPossessorTeammate = cv::norm(m_BallPossessorCoordinates - playerHomePos); // Benefit
			if (fDistPossessorTeammate > MAX_DISTANCE_TO_PASS)
			{
				continue;
			}

			float fDistBallSegmentAwayTeam = std::numeric_limits<double>::max();
			// Take the minimum distance between segment ball (from possessor to current teammate)
			for (const cv::Point2i& playersAwayPos : awayPlayers) // Success
			{
				Line<cv::Point2f> line(m_BallPossessorCoordinates, playerHomePos);
				float fCurrDist = line.GetDistanceWith(playersAwayPos);
				if (fCurrDist < fDistBallSegmentAwayTeam)
				{
					fDistBallSegmentAwayTeam = fCurrDist;
				}
			}
			
			float fCurrProbability = /*benefit*/InterpolateProbabilitiesBetweenIntervals(fDistPossessorTeammate, MIN_DISTANCE_TO_PASS, MAX_DISTANCE_TO_PASS,
				MIN_PROBABILITY, MAX_PROBABILITY) * /*success*/ InterpolateProbabilitiesBetweenIntervals(fDistBallSegmentAwayTeam,
					MIN_DISTANCE_FROM_AWAY, MAX_DISTANCE_FROM_AWAY, MIN_PROBABILITY, MAX_PROBABILITY);

			if (fCurrProbability > fProbability)
			{
				fProbability = fCurrProbability;
				m_PlayerToPass = cv::Vec2f(playerHomePos.x - m_BallPossessorCoordinates.x, playerHomePos.y - m_BallPossessorCoordinates.y);
			}
		}

		return fProbability;
	}

	return 1 / (BEHAVIORS_NUM + 100); // We don't have the ball
}

// TODO!!!
float UtilityAI::TackleBehavior(ComputerVisionLayer::VisionOutput& _computerVisionOutput)
{
	if (m_bIsOffensiveTeam)
	{
		return 1 / (BEHAVIORS_NUM + 100); // We have the ball
	}

	// We are in defensive
	float fAngle = AngleBetween(m_BallDefensiveHolder, m_BallPossessorCoordinates);
	float fDist = (float)cv::norm(m_BallDefensiveHolder - m_BallPossessorCoordinates);

	//float fBenefit = //float fSuccess =
	// TODO: define a distance!!! and depending on it return the type of tackle
	m_PlayerToTackle = cv::Vec2f(m_BallPossessorCoordinates.x - m_BallDefensiveHolder.x, m_BallPossessorCoordinates.y - m_BallDefensiveHolder.y);

	return  (fAngle / 360.0f) * (1.0f / fDist);
}
#define MIN_ANGLE_TO_SHOOT 0.0f
#define MAX_ANGLE_TO_SHOOT 35.0f
#define MIN_DIST_TO_SHOOT 10.0f
#define MAX_DIST_TO_SHOOT 300.0f

#define MAX_PROBABILITY_SHOOT_GOAL_NOT_VISIBLE 0.5f

float UtilityAI::ShootBehavior(ComputerVisionLayer::VisionOutput& _computerVisionOutput)
{
	if (m_bIsOffensiveTeam)
	{
		cv::Point2f goalPosition;
		if (_computerVisionOutput.m_isGoalVisible)
		{
			if (_computerVisionOutput.m_isRealCoordinates)
			{
				goalPosition = _computerVisionOutput.m_goalPos_real;
			}
			else
			{
				goalPosition = _computerVisionOutput.m_goalPos_img;
			}

			float fDistGoalAndPossessor = cv::norm((cv::Point2f)m_BallPossessorCoordinates - goalPosition);
			float fAngle = AngleBetween(m_BallPossessorCoordinates, goalPosition);

			if (fAngle <= MAX_ANGLE_TO_SHOOT && fDistGoalAndPossessor <= MAX_DIST_TO_SHOOT) // TODO : USE DISTANCE OR NOT??? and away team posibilities to attack
			{
				m_PositionToShoot = goalPosition;
				return InterpolateProbabilitiesBetweenIntervals(fAngle, MIN_ANGLE_TO_SHOOT, MAX_ANGLE_TO_SHOOT, MIN_PROBABILITY, MAX_PROBABILITY);
			}
		}
		else // if the goal is not visible
		{
			std::vector<ComputerVisionLayer::KeyPointData> keyPointData;
			if (_computerVisionOutput.m_isRealCoordinates)
			{
				keyPointData = _computerVisionOutput.m_keyPointsOutput_real;
			}
			else
			{
				keyPointData = _computerVisionOutput.m_keyPointsOutput_img;
			}

			// Save small box up and down coordinates if there are in output
			// Also save box up and down coordinates if there are in output
			cv::Vec2f nullVec2f = cv::Vec2f(0.0f, 0.0f);
			cv::Vec2f smallBoxUp = nullVec2f, smallBoxDown = nullVec2f;
			cv::Vec2f boxUp = nullVec2f, boxDown = nullVec2f;
			
			for (const ComputerVisionLayer::KeyPointData& keyPoint : keyPointData)
			{
				if (keyPoint.first == ComputerVisionLayer::PitchKeyPoints::FEATURE_SMALL_BOX_UP)
				{
					smallBoxUp = keyPoint.second;
				}
				else if(keyPoint.first == ComputerVisionLayer::PitchKeyPoints::FEATURE_SMALL_BOX_DOWN)
				{
					smallBoxDown = keyPoint.second;
				}
				else if (keyPoint.first == ComputerVisionLayer::PitchKeyPoints::FEATURE_BOX_UP)
				{
					boxUp = keyPoint.second;
				}
				else if (keyPoint.first == ComputerVisionLayer::PitchKeyPoints::FEATURE_BOX_DOWN)
				{
					boxDown = keyPoint.second;
				}
			}

			// First we look to small box up and small box down if there are visible
			if (smallBoxUp != nullVec2f && smallBoxDown != nullVec2f)
			{
				// We should estimate the gate position!!
				m_PositionToShoot = (smallBoxDown - smallBoxUp) / 2 + cv::Vec2f((_computerVisionOutput.m_fieldSideToAttack * 100.0f), 0.0f);
			}
			else if (boxUp != nullVec2f && boxDown != nullVec2f) // If not, we look for the large boxes
			{
				// We should estimate the gate position!!
				m_PositionToShoot = (boxDown - boxUp) / 2 + cv::Vec2f((_computerVisionOutput.m_fieldSideToAttack * 100.0f), 0.0f);
			}
			else if (boxUp != nullVec2f && smallBoxUp != nullVec2f)
			{
				m_PositionToShoot = boxUp + (smallBoxUp - boxUp) * 2.0f;
			}
			else if (boxDown != nullVec2f && smallBoxDown != nullVec2f)
			{
				m_PositionToShoot = boxDown + (smallBoxDown - boxDown) * 2.0f;
			}

			float fAngle = AngleBetween(m_BallPossessorCoordinates, m_PositionToShoot);

			return InterpolateProbabilitiesBetweenIntervals(fAngle, MIN_ANGLE_TO_SHOOT, MAX_ANGLE_TO_SHOOT, MIN_PROBABILITY, MAX_PROBABILITY_SHOOT_GOAL_NOT_VISIBLE);
		}
	}

	return 1 / (BEHAVIORS_NUM + 100); // We don't have the ball or is impossible to shoot
}

#define DRIBBLE_ANGLE 30.0f
#define FIRST_DISTANCE_TO_DRIBBLE 50.0f
#define SECOND_DISTANCE_TO_DRIBBLE 100.0f
#define MIN_DIST_TO_AWAY_TEAM 0.0f
#define MAX_DIST_TO_AWAY_TEAM 100.0f
#define NEUTRAL_PROBABILITY 0.5f

void Rotate(cv::Vec2f& _vecToRotate)
{
	_vecToRotate[0] = _vecToRotate[0] * cos(DRIBBLE_ANGLE) - _vecToRotate[1] * sin(DRIBBLE_ANGLE);
	_vecToRotate[1] = _vecToRotate[0] * sin(DRIBBLE_ANGLE) + _vecToRotate[1] * cos(DRIBBLE_ANGLE);
}

float UtilityAI::DribbleBehavior(ComputerVisionLayer::VisionOutput& _computerVisionOutput)
{
	if (m_bIsOffensiveTeam)
	{
		vector<std::pair<cv::Point2f, float>> firstDistToDribble;
		vector<std::pair<cv::Point2f, float>> secondDistToDribble;

		cv::Vec2f firstDistStart = cv::Vec2f(m_BallPossessorCoordinates.x + FIRST_DISTANCE_TO_DRIBBLE, m_BallPossessorCoordinates.y);
		cv::Vec2f secondDistStart = cv::Vec2f(m_BallPossessorCoordinates.x + SECOND_DISTANCE_TO_DRIBBLE, m_BallPossessorCoordinates.y);

		for (int i = 0; i < 360 / DRIBBLE_ANGLE; i++)
		{
			firstDistToDribble.push_back(std::make_pair(firstDistStart, 0.0f));
			secondDistToDribble.push_back(std::make_pair(secondDistStart, 0.0f));

			Rotate(firstDistStart);
			Rotate(secondDistStart);
		}

		std::vector<ComputerVisionLayer::VisionOutput::PlayerInfo> awayPlayers;

		if (_computerVisionOutput.m_isRealCoordinates)
		{
			awayPlayers = _computerVisionOutput.m_awayPlayers_real;
		}
		else
		{
			awayPlayers = _computerVisionOutput.m_awayPlayers_img;
		}

		float fProbability = std::numeric_limits<double>::min();

		for (std::pair<cv::Point2f, float>& dribblePoint : firstDistToDribble)
		{
			float fMinDistDribblePointToAwayTeam = std::numeric_limits<double>::max();
			for (const cv::Point2i& playerAwayPos : awayPlayers)
			{
				Line<cv::Point2f> line((cv::Point2f)m_BallPossessorCoordinates, dribblePoint.first);
				float fCurrDist = line.GetDistanceWith(playerAwayPos);
				if (fCurrDist < fMinDistDribblePointToAwayTeam)
				{
					fMinDistDribblePointToAwayTeam = fCurrDist;
				}
			}

			dribblePoint.second = /*Benefit*/InterpolateProbabilitiesBetweenIntervals(FIRST_DISTANCE_TO_DRIBBLE, FIRST_DISTANCE_TO_DRIBBLE, SECOND_DISTANCE_TO_DRIBBLE, NEUTRAL_PROBABILITY, MAX_PROBABILITY)*
				/*Success*/InterpolateProbabilitiesBetweenIntervals(fMinDistDribblePointToAwayTeam, FIRST_DISTANCE_TO_DRIBBLE, SECOND_DISTANCE_TO_DRIBBLE, NEUTRAL_PROBABILITY, MAX_PROBABILITY);

			if (dribblePoint.second > fProbability)
			{
				fProbability = dribblePoint.second;
				m_PositionToDribble = dribblePoint.first;
			}
		}

		for (std::pair<cv::Point2f, float>& dribblePoint : secondDistToDribble)
		{
			float fMinDistDribblePointToAwayTeam = std::numeric_limits<double>::max();
			for (const cv::Point2i& playerAwayPos : awayPlayers)
			{
				Line<cv::Point2f> line((cv::Point2f)m_BallPossessorCoordinates, dribblePoint.first);
				float fCurrDist = line.GetDistanceWith(playerAwayPos);
				if (fCurrDist < fMinDistDribblePointToAwayTeam)
				{
					fMinDistDribblePointToAwayTeam = fCurrDist;
				}
			}

			dribblePoint.second = /*Benefit*/InterpolateProbabilitiesBetweenIntervals(FIRST_DISTANCE_TO_DRIBBLE, FIRST_DISTANCE_TO_DRIBBLE, SECOND_DISTANCE_TO_DRIBBLE, NEUTRAL_PROBABILITY, MAX_PROBABILITY)*
				/*Success*/InterpolateProbabilitiesBetweenIntervals(fMinDistDribblePointToAwayTeam, FIRST_DISTANCE_TO_DRIBBLE, SECOND_DISTANCE_TO_DRIBBLE, NEUTRAL_PROBABILITY, MAX_PROBABILITY);
			
			if (dribblePoint.second > fProbability)
			{
				fProbability = dribblePoint.second;
				m_PositionToDribble = dribblePoint.first;
			}
		}

		return fProbability;
	}

	return 1 / (BEHAVIORS_NUM + 100);  // We don't have the ball
}
