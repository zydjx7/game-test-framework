#include "InputSendingLayer.h"
#include <iostream>
using namespace std;

InputSendingLayerImpl::InputSendingLayerImpl()
{
	m_ActionsProcessed.reset();
}

// Convert Packet type to String
BOOL PacketType2Str(FFBPType Type, LPTSTR OutStr)
{
	BOOL stat = TRUE;
	LPTSTR Str = "";

	switch (Type)
	{
	case PT_EFFREP:
		Str = "Effect Report";
		break;
	case PT_ENVREP:
		Str = "Envelope Report";
		break;
	case PT_CONDREP:
		Str = "Condition Report";
		break;
	case PT_PRIDREP:
		Str = "Periodic Report";
		break;
	case PT_CONSTREP:
		Str = "Constant Force Report";
		break;
	case PT_RAMPREP:
		Str = "Ramp Force Report";
		break;
	case PT_CSTMREP:
		Str = "Custom Force Data Report";
		break;
	case PT_SMPLREP:
		Str = "Download Force Sample";
		break;
	case PT_EFOPREP:
		Str = "Effect Operation Report";
		break;
	case PT_BLKFRREP:
		Str = "PID Block Free Report";
		break;
	case PT_CTRLREP:
		Str = "PID Device Contro";
		break;
	case PT_GAINREP:
		Str = "Device Gain Report";
		break;
	case PT_SETCREP:
		Str = "Set Custom Force Report";
		break;
	case PT_NEWEFREP:
		Str = "Create New Effect Report";
		break;
	case PT_BLKLDREP:
		Str = "Block Load Report";
		break;
	case PT_POOLREP:
		Str = "PID Pool Report";
		break;
	default:
		stat = FALSE;
		break;
	}

	if (stat)
		_tcscpy_s(OutStr, 100, Str);

	return stat;
}

// Convert Effect type to String
BOOL /*InputSendingLayerImpl::*/EffectType2Str(FFBEType Type, LPTSTR OutStr)
{
	BOOL stat = TRUE;
	LPTSTR Str = "";

	switch (Type)
	{
	case ET_NONE:
		stat = FALSE;
		break;
	case ET_CONST:
		Str = "Constant Force";
		break;
	case ET_RAMP:
		Str = "Ramp";
		break;
	case ET_SQR:
		Str = "Square";
		break;
	case ET_SINE:
		Str = "Sine";
		break;
	case ET_TRNGL:
		Str = "Triangle";
		break;
	case ET_STUP:
		Str = "Sawtooth Up";
		break;
	case ET_STDN:
		Str = "Sawtooth Down";
		break;
	case ET_SPRNG:
		Str = "Spring";
		break;
	case ET_DMPR:
		Str = "Damper";
		break;
	case ET_INRT:
		Str = "Inertia";
		break;
	case ET_FRCTN:
		Str = "Friction";
		break;
	case ET_CSTM:
		Str = "Custom Force";
		break;
	default:
		stat = FALSE;
		break;
	};

	if (stat)
		_tcscpy_s(OutStr, 100, Str);

	return stat;
}

// Convert PID Device Control to String
BOOL DevCtrl2Str(FFB_CTRL Ctrl, LPTSTR OutStr)
{
	BOOL stat = TRUE;
	LPTSTR Str = "";

	switch (Ctrl)
	{
	case CTRL_ENACT:
		Str = "Enable Actuators";
		break;
	case CTRL_DISACT:
		Str = "Disable Actuators";
		break;
	case CTRL_STOPALL:
		Str = "Stop All Effects";
		break;
	case CTRL_DEVRST:
		Str = "Device Reset";
		break;
	case CTRL_DEVPAUSE:
		Str = "Device Pause";
		break;
	case CTRL_DEVCONT:
		Str = "Device Continue";
		break;
	default:
		stat = FALSE;
		break;
	}
	if (stat)
		_tcscpy_s(OutStr, 100, Str);

	return stat;
}

// Convert Effect operation to string
BOOL EffectOpStr(FFBOP Op, LPTSTR OutStr)
{
	BOOL stat = TRUE;
	LPTSTR Str = "";

	switch (Op)
	{
	case EFF_START:
		Str = "Effect Start";
		break;
	case EFF_SOLO:
		Str = "Effect Solo Start";
		break;
	case EFF_STOP:
		Str = "Effect Stop";
		break;
	default:
		stat = FALSE;
		break;
	}

	if (stat)
		_tcscpy_s(OutStr, 100, Str);

	return stat;
}

// Polar values (0x00-0xFF) to Degrees (0-360)
int Polar2Deg(BYTE Polar)
{
	return ((UINT)Polar * 360) / 255;
}

// Convert range 0x00-0xFF to 0%-100%
int Byte2Percent(BYTE InByte)
{
	return ((UINT)InByte * 100) / 255;
}

// Convert One-Byte 2's complement input to integer
int TwosCompByte2Int(BYTE in)
{
	int tmp;
	BYTE inv = ~in;
	BOOL isNeg = in >> 7;
	if (isNeg)
	{
		tmp = (int)(inv);
		tmp = -1 * tmp;
		return tmp;
	}
	else
		return (int)in;
}

// Generic callback function
void CALLBACK FfbFunction(PVOID data)
{
	FFB_DATA * FfbData = (FFB_DATA *)data;
	int size = FfbData->size;
	_tprintf("\nFFB Size %d\n", size);

	_tprintf("Cmd:%08.8X ", FfbData->cmd);
	_tprintf("ID:%02.2X ", FfbData->data[0]);
	_tprintf("Size:%02.2d ", static_cast<int>(FfbData->size - 8));
	_tprintf(" - ");
	for (UINT i = 0; i < FfbData->size - 8; i++)
		_tprintf(" %02.2X", (UINT)FfbData->data);
	_tprintf("\n");
}

void CALLBACK FfbFunction1(PVOID data, PVOID userdata)
{
	// Packet Header
	_tprintf("\n ============= FFB Packet size Size %d =============\n", static_cast<int>(((FFB_DATA *)data)->size));

	/////// Packet Device ID, and Type Block Index (if exists)
#pragma region Packet Device ID, and Type Block Index
	int DeviceID, BlockIndex;
	FFBPType	Type;
	TCHAR	TypeStr[100];

	if (ERROR_SUCCESS == Ffb_h_DeviceID((FFB_DATA *)data, &DeviceID))
		_tprintf("\n > Device ID: %d", DeviceID);
	if (ERROR_SUCCESS == Ffb_h_Type((FFB_DATA *)data, &Type))
	{
		if (!PacketType2Str(Type, TypeStr))
			_tprintf("\n > Packet Type: %d", Type);
		else
			_tprintf("\n > Packet Type: %s", TypeStr);

	}
	if (ERROR_SUCCESS == Ffb_h_EBI((FFB_DATA *)data, &BlockIndex))
		_tprintf("\n > Effect Block Index: %d", BlockIndex);
#pragma endregion


	/////// Effect Report
#pragma region Effect Report
	FFB_EFF_CONST Effect;
	if (ERROR_SUCCESS == Ffb_h_Eff_Report((FFB_DATA *)data, &Effect))
	{
		if (!EffectType2Str(Effect.EffectType, TypeStr))
			_tprintf("\n >> Effect Report: %02x", Effect.EffectType);
		else
			_tprintf("\n >> Effect Report: %s", TypeStr);

		if (Effect.Polar)
		{
			_tprintf("\n >> Direction: %d deg (%02x)", Polar2Deg(Effect.Direction), Effect.Direction);


		}
		else
		{
			_tprintf("\n >> X Direction: %02x", Effect.DirX);
			_tprintf("\n >> Y Direction: %02x", Effect.DirY);
		};

		if (Effect.Duration == 0xFFFF)
			_tprintf("\n >> Duration: Infinit");
		else
			_tprintf("\n >> Duration: %d MilliSec", static_cast<int>(Effect.Duration));

		if (Effect.TrigerRpt == 0xFFFF)
			_tprintf("\n >> Trigger Repeat: Infinit");
		else
			_tprintf("\n >> Trigger Repeat: %d", static_cast<int>(Effect.TrigerRpt));

		if (Effect.SamplePrd == 0xFFFF)
			_tprintf("\n >> Sample Period: Infinit");
		else
			_tprintf("\n >> Sample Period: %d", static_cast<int>(Effect.SamplePrd));


		_tprintf("\n >> Gain: %d%%", Byte2Percent(Effect.Gain));

	};
#pragma endregion
#pragma region PID Device Control
	FFB_CTRL	Control;
	TCHAR	CtrlStr[100];
	if (ERROR_SUCCESS == Ffb_h_DevCtrl((FFB_DATA *)data, &Control) && DevCtrl2Str(Control, CtrlStr))
		_tprintf("\n >> PID Device Control: %s", CtrlStr);

#pragma endregion
#pragma region Effect Operation
	FFB_EFF_OP	Operation;
	TCHAR	EffOpStr[100];
	if (ERROR_SUCCESS == Ffb_h_EffOp((FFB_DATA *)data, &Operation) && EffectOpStr(Operation.EffectOp, EffOpStr))
	{
		_tprintf("\n >> Effect Operation: %s", EffOpStr);
		if (Operation.LoopCount == 0xFF)
			_tprintf("\n >> Loop until stopped");
		else
			_tprintf("\n >> Loop %d times", static_cast<int>(Operation.LoopCount));

	};
#pragma endregion
#pragma region Global Device Gain
	BYTE Gain;
	if (ERROR_SUCCESS == Ffb_h_DevGain((FFB_DATA *)data, &Gain))
		_tprintf("\n >> Global Device Gain: %d", Byte2Percent(Gain));

#pragma endregion
#pragma region Condition
	FFB_EFF_COND Condition;
	if (ERROR_SUCCESS == Ffb_h_Eff_Cond((FFB_DATA *)data, &Condition))
	{
		if (Condition.isY)
			_tprintf("\n >> Y Axis");
		else
			_tprintf("\n >> X Axis");
		_tprintf("\n >> Center Point Offset: %d", TwosCompByte2Int(Condition.CenterPointOffset) * 10000 / 127);
		_tprintf("\n >> Positive Coefficient: %d", TwosCompByte2Int(Condition.PosCoeff) * 10000 / 127);
		_tprintf("\n >> Negative Coefficient: %d", TwosCompByte2Int(Condition.NegCoeff) * 10000 / 127);
		_tprintf("\n >> Positive Saturation: %d", Condition.PosSatur * 10000 / 255);
		_tprintf("\n >> Negative Saturation: %d", Condition.NegSatur * 10000 / 255);
		_tprintf("\n >> Dead Band: %d", Condition.DeadBand * 10000 / 255);
	}
#pragma endregion
#pragma region Envelope
	FFB_EFF_ENVLP Envelope;
	if (ERROR_SUCCESS == Ffb_h_Eff_Envlp((FFB_DATA *)data, &Envelope))
	{
		_tprintf("\n >> Attack Level: %d", Envelope.AttackLevel * 10000 / 255);
		_tprintf("\n >> Fade Level: %d", Envelope.FadeLevel * 10000 / 255);
		_tprintf("\n >> Attack Time: %d", static_cast<int>(Envelope.AttackTime));
		_tprintf("\n >> Fade Time: %d", static_cast<int>(Envelope.FadeTime));
	};

#pragma endregion
#pragma region Periodic
	FFB_EFF_PERIOD EffPrd;
	if (ERROR_SUCCESS == Ffb_h_Eff_Period((FFB_DATA *)data, &EffPrd))
	{
		_tprintf("\n >> Magnitude: %d", EffPrd.Magnitude * 10000 / 255);
		_tprintf("\n >> Offset: %d", TwosCompByte2Int(EffPrd.Offset) * 10000 / 127);
		_tprintf("\n >> Phase: %d", EffPrd.Phase * 3600 / 255);
		_tprintf("\n >> Period: %d", static_cast<int>(EffPrd.Period));
	};
#pragma endregion

#pragma region Effect Type
	FFBEType EffectType;
	if (ERROR_SUCCESS == Ffb_h_EffNew((FFB_DATA *)data, &EffectType))
	{
		if (EffectType2Str(EffectType, TypeStr))
			_tprintf("\n >> Effect Type: %s", TypeStr);
		else
			_tprintf("\n >> Effect Type: Unknown");
	}

#pragma endregion

#pragma region Ramp Effect
	FFB_EFF_RAMP RampEffect;
	if (ERROR_SUCCESS == Ffb_h_Eff_Ramp((FFB_DATA *)data, &RampEffect))
	{
		_tprintf("\n >> Ramp Start: %d", TwosCompByte2Int(RampEffect.Start) * 10000 / 127);
		_tprintf("\n >> Ramp End: %d", TwosCompByte2Int(RampEffect.End) * 10000 / 127);
	};

#pragma endregion

	_tprintf("\n");
	FfbFunction(data);
	_tprintf("\n ====================================================\n");

}

void InputSendingLayerImpl::init()
{
	const UINT DevID = DEV_ID;

	// Get the driver attributes (Vendor ID, Product ID, Version Number)
	if (!vJoyEnabled())
	{
		//_tprintf("Function vJoyEnabled Failed - make sure that vJoy is installed and enabled\n");
		RelinquishVJD(DevID);
		return;
	}
	//PrintVJoyStatus(DevID);
	// Acquire the vJoy device
	if (!AcquireVJD(DevID))
	{
		//_tprintf("Failed to acquire vJoy device number %d.\n", DevID);
		RelinquishVJD(DevID);
		return;
	}
	BOOL Ffbstarted = FfbStart(DevID);
	if (!Ffbstarted)
	{
		//_tprintf("Failed to start FFB on vJoy device number %d.\n", DevID);
		RelinquishVJD(DevID);
		return;
	}
	FfbRegisterGenCB(FfbFunction1, NULL);
}

void InputSendingLayerImpl::ProcessActions(Actions& _actionsToDo)
{
	const UINT DevID = DEV_ID;

	JOYSTICK_POSITION_V2 iReport = GetJoystickPositionData();

	// Set destenition vJoy device
	BYTE id = (BYTE)DevID;
	iReport.bDevice = id;

	// DEbuging...
	//ResetStruct(_actionsToDo);
	
	
	// TODO: m_StickDir - when we receive the structure we will update the iReport.wAxisY and iReport.wAxisX with values from m_StickDir
	// TODO: we should update always the stick 
	if (_actionsToDo.bDribble)
	{
		m_ActionsProcessed.fStickAngle = _actionsToDo.fStickAngle;
		AngleToHexaValues(m_ActionsProcessed.fStickAngle, iReport.wAxisX, iReport.wAxisY);
		m_ActionsProcessed.bDribble = true;
	}
	else //if (m_ActionsProcessed.bDribble) // Reset stick if no action like dribble
	{
		iReport.wAxisX = NEUTRAL_AXIS_HEX_VALUE;
		iReport.wAxisY = NEUTRAL_AXIS_HEX_VALUE;
		iReport.wAxisZ = NEUTRAL_AXIS_HEX_VALUE;
		m_ActionsProcessed.bDribble = false;
		m_ActionsProcessed.fStickAngle = 0.0f;
	}

	// m_bShoot -> shoot -> B
	if (_actionsToDo.bShoot)// Want to Start the input
	{
		iReport.lButtons = 2; 
		m_ActionsProcessed.bShoot = true;
	}
	else //if (m_ActionsProcessed.bShoot) // Want to Stop the- input
	{
		iReport.lButtons = 0;
		m_ActionsProcessed.bShoot = false;
	}

	// LB + B ???
	if (_actionsToDo.bShootChip)
	{
		if (!m_ActionsProcessed.bShootChip)
		{
			iReport.lButtonsEx1 = 40; // LB???
			iReport.lButtons = 2;
			m_ActionsProcessed.bShootChip = true;
		}
	}
	else //if (m_ActionsProcessed.bShootChip)
	{
		iReport.lButtonsEx1 = 0; // LB???
		iReport.lButtons = 0;
		m_ActionsProcessed.bShootChip = false;
	}

	// B + B
	if (_actionsToDo.bShootDriven)
	{
		if (!m_ActionsProcessed.bShootDriven)
		{
			iReport.lButtons = 2;
			m_ActionsProcessed.bShootDriven = true;
			
			SendPositionDataToVJoyDevice(DevID, iReport); // Send here one more time because we should press 2xB for shoot driven
		}
	}
	else //if (m_ActionsProcessed.bShootDriven)
	{
		iReport.lButtons = 0;
		m_ActionsProcessed.bShootDriven = false;
	}

	// m_bPass -> pass -> A
	if (_actionsToDo.bPass) // Want to Start the input
	{
		iReport.lButtons = 4; 
		m_ActionsProcessed.bPass = true;
	}
	else //if (m_ActionsProcessed.bPass) // Want to Stop the input
	{
		iReport.lButtons = 0;
		m_ActionsProcessed.bPass = false;
	}

	// m_bCross -> X 
	if (_actionsToDo.bTackleSlide)
	{
		m_ActionsProcessed.bTackleSlide = true;
		iReport.lButtons = 8; 
	}
	else //if (m_ActionsProcessed.bTackleSlide)
	{
		iReport.lButtons = 0; 
		m_ActionsProcessed.bTackleSlide = false;
	}

	// bSprint -> RT
	if (_actionsToDo.bSprint)
	{
		m_ActionsProcessed.bSprint = true;
		iReport.lButtons = 20; 
	}
	else //if (m_ActionsProcessed.bSprint)
	{
		iReport.lButtons = 0; 
		m_ActionsProcessed.bSprint = false;
	}

	SendPositionDataToVJoyDevice(DevID, iReport);
}

// function to convert decimal to hexadecimal
void InputSendingLayerImpl::DecimalToHexa(int _value, LONG &_outHexNum)
{
	// char array to store hexadecimal number
	char hexaDeciNum[100];
	char *pEnd;
	itoa(_value, hexaDeciNum, 16);
	_outHexNum = strtoll(hexaDeciNum, NULL, 0);
	//sscanf(hexaDeciNum, "%X", &_outHexNum);
}

#define PI 3.141592653589793
float DegreeToRadian(float _angle)
{
	return _angle * PI / 180.0f;
}

void InputSendingLayerImpl::AngleToHexaValues(float _angle, LONG &_xValue, LONG &_yValue)
{
	// Depending on what we received we should call cos and sin with the angle in radian!!!
	int radianAngle = DegreeToRadian(_angle);
	float X = cos(radianAngle);
	float Y = sin(radianAngle);

	float xAxis = MIN_AXIS_VALUE + (X + 1) * (MAX_AXIS_VALUE - MIN_AXIS_VALUE) / 2;
	float yAxis = MIN_AXIS_VALUE + (Y + 1) * (MAX_AXIS_VALUE - MIN_AXIS_VALUE) / 2;

	DecimalToHexa(xAxis, _xValue);
	DecimalToHexa(yAxis, _yValue);
}

void InputSendingLayerImpl::PrintVJoyStatus(UINT _devID)
{
	// Get the status of the vJoy device before trying to acquire it
	VjdStat status = GetVJDStatus(_devID);

	switch (status)
	{
	case VJD_STAT_OWN:
		_tprintf("vJoy device %d is already owned by this feeder\n", _devID);
		break;
	case VJD_STAT_FREE:
		_tprintf("vJoy device %d is free\n", _devID);
		break;
	case VJD_STAT_BUSY:
		_tprintf("vJoy device %d is already owned by another feeder\nCannot continue\n", _devID);
		break;
	case VJD_STAT_MISS:
		_tprintf("vJoy device %d is not installed or disabled\nCannot continue\n", _devID);
		break;
	default:
		_tprintf("vJoy device %d general error\nCannot continue\n", _devID);
		break;
	};

}

void InputSendingLayerImpl::SendPositionDataToVJoyDevice(UINT _devID, JOYSTICK_POSITION_V2 &_iReport)
{
	// Send position data to vJoy device
	PVOID pPositionMessage = (PVOID)(&_iReport);
	if (!UpdateVJD(_devID, pPositionMessage))
	{
		AcquireVJD(_devID);
	}
}
