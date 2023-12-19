from Raspi_PWM_Servo_Driver import PWM
from Raspi_MotorHAT import Raspi_MotorHAT, Raspi_DCMotor
from time import sleep
#from sense_hat import SenseHat

class Signal_move:
    def __init__(self):
        self.motor_addr = 0x6f
        self.nfm_motor = 2
        self.turn_speed = 100
        self.straight_speed = 60
        
        self.servo_addr = 0x6F
        self.servo_freq = 50
        self.servo_pin = 14
        self.servo_default = [310, 425, 500]
        
        self.mh = Raspi_MotorHAT(addr=0x6f)
        self.myMotor = self.mh.getMotor(2)
        self.servo_pwm = PWM(0x6F)
        self.servo_pwm.setPWMFreq(50)
        
        '''
            myMotor.setSpeed(150)
            myMotor.run(Raspi_MotorHAT.FORWARD)
            myMotor.run(Raspi_MotorHAT.BACKWARD)
            myMotor.run(Raspi_MotorHAT.RELEASE)
            
            servo_pwm.setPWM(servo_pin, 0, servo_default[1])
        '''
        
    '''
        입력을 받는 경우와 아닌 경우
        - 입력을 받을 경우에는 테스트중인 상황으로
            계속해서 사용자의 입력을 받아야한다.
        - 아닌 경우는 함수를 호출 하는 경우
    '''
    
    def trun_angle(self, interval_time, servo_value):
        self.servo_pwm.setPWM(self.servo_pin, 0, servo_value)
        #sleep(0.5)
        
        self.myMotor.setSpeed(self.turn_speed)
        self.myMotor.run(Raspi_MotorHAT.FORWARD)
        sleep(interval_time)
       
        
    def turn_left_(self, interval_time):
        self.servo_pwm.setPWM(self.servo_pin, 0, self.servo_default[0])
        #sleep(0.5)
        
        self.myMotor.setSpeed(self.turn_speed)
        self.myMotor.run(Raspi_MotorHAT.FORWARD)
        sleep(interval_time)
        
        
    def turn_left(self, interval_time, servo_value):
        self.trun_angle(interval_time, servo_value)
        
        
    def turn_right_(self, interval_time):
        self.servo_pwm.setPWM(self.servo_pin, 0, self.servo_default[2])
        #sleep(0.5)
        
        self.myMotor.setSpeed(self.turn_speed)
        self.myMotor.run(Raspi_MotorHAT.FORWARD)
        sleep(interval_time)
        
        
    def turn_right(self, interval_time, servo_value):
        self.trun_angle(interval_time, servo_value)
        
    
    def go_straight(self, interval_time):
        self.servo_pwm.setPWM(self.servo_pin, 0, self.servo_default[1])
        
        self.myMotor.setSpeed(self.straight_speed)
        self.myMotor.run(Raspi_MotorHAT.FORWARD)
        sleep(interval_time)
        
    def stop_move(self):
        self.myMotor.setSpeed(0)
        self.myMotor.run(Raspi_MotorHAT.FORWARD)
    
    def go_back(self, interval_time):
        self.myMotor.setSpeed(self.turn_speed)
        self.myMotor.run(Raspi_MotorHAT.BACKWARD)

        sleep(interval_time)



def is_inRange(sv):
    return sv >= 310 and sv <= 500

#for test
if __name__ == "__main__":
    signal_obj = Signal_move()
    input_time = 0
    
    while True:
        signal_obj.stop_move()
        cmd = int(input("Enter command\n[right : 1] [left : 2] [go straight : 3] [stop : 4] [back : 9] [exit : 0]\n::"))
        
        if cmd == 1:#right
            input_time, input_value = input('Enter time, servo\'s value[default : 0]: ').split()
            if(int(input_value) == 0):
                signal_obj.turn_right_(float(input_time))
            else:
                if is_inRange(int(input_value)):
                    signal_obj.turn_right(float(input_time), int(input_value))
                else:
                    print('Wrong input servo value!!!')
        elif cmd == 2:#left
            input_time, input_value = input('Enter time, servo\'s value[default : 0]: ').split()
            if(int(input_value) == 0):
                signal_obj.turn_left_(float(input_time))
            else:
                if is_inRange(int(input_value)):
                    signal_obj.turn_left(float(input_time), int(input_value))
                else:
                    print('Wrong input servo value!!!')
        elif cmd == 3:#go straight
            input_time = float(input('Enter time: '))
            signal_obj.go_straight(input_time)
        elif cmd == 4:#stop
            signal_obj.stop_move()
        elif cmd == 9:#go back
            signal_obj.go_back(float(input_time))
        elif cmd == 0:#exit
            break
        else:#wrong cmd
            print("Entered Wrong cmd")
                                
            
