#!/usr/bin/env python3
######## can REPLACE later:
# import rclpy
# from rclpy.node import Node
# from rosidl_generator_cpp import msg_type_to_cpp
# from std_msgs.msg import String
#
# class GPSDriver(Node):
#     def __init__(self):
#         super().__init__('gps_driver')
#         self.publisher_ = self.create_publisher(String, 'gps_topic', 10)
#         timer_period = 1.0
#         self.timer = self.create_timer(timer_period, self.timer_callback)
#
#     def timer_callback(self):
#         msg = String()
#         msg.data = 'Hello from GPS driver'
#         self.publisher_.publish(msg)
#         self.get_logger().info(f'Published: "{msg.data}"')
#
# def main(args=None):
#     rclpy.init(args=args)
#     node = GPSDriver()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#
# if __name__ == '__main__':
#     main()

#####################################################SSS
import rclpy
from rclpy.node import Node
from rclpy.utilities import timeout_sec_to_nsec
from rqt_bag.plugins.raw_view import YAW_LABEL
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, MagneticField
#from gps_driver.msg import Customgps
from custom_interfaces.msg import Vectornav
from scipy.spatial.transform import Rotation as R
##################################################EEE

import utm
import time
import serial

#################SSSSSSSSSSSSSSSSS
#gpggaRead1 = '$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,TSTR*61'
# $VNYMR,Yaw,Pitch,Roll,MagX,MagY,MagZ,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ*CS
class VNDriver(Node):
    def __init__(self):
        super().__init__('vn_driver')
        print('init')
        #initialize publisher
        self.publisher_ = self.create_publisher(Vectornav, '/imu', 10) # first input is .msg file w/o ".msg"
        self.timer = self.create_timer(0.01, self.timer_callback)


        #serial setup
        #serial_port_address = '/dev/pts/10' #####CHANGE as needed to usb port!!
        #serialPortAddr = '/dev/ttyUSB0' #'/dev/pts/4'  # will need to be replaced with the serial port address. OK to hardcode for
        # now, but eventually your launch file should handle this

        #Update for launch file! with port
        self.declare_parameter('port', '/dev/ttyUSB0')
        serialPortAddr = self.get_parameter('port').get_parameter_value().string_value
        self.get_logger().info(f'Serial Port: {serialPortAddr}')

        try:
            self.serialPort = serial.Serial(serialPortAddr, 115200, timeout=0.1)  # This line opens the port
            self.get_logger().info(f'Serial port {serialPortAddr} opened.')
            print('opened')

            change_hz_write = "$VNWRG,07,40*59"
            self.serialPort.write(change_hz_write.encode('utf-8'))
            print('write to 40 hz')

        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {serialPortAddr}')
            self.serialPort = None

    def timer_callback(self):
        print('timercallback')
        if self.serialPort.in_waiting > 0:
            print('hellooo')
            vnymr_line = self.serialPort.readline().decode('utf-8', errors='ignore')
            if 'VNYMR' in vnymr_line:
                print('vnymr')
                msg =self.parse_vnymr(vnymr_line)
                if msg:
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published: {msg}')


###############EEEEEEEEEEEE

#Stuff you will do only when the script is first run:
#serialPortAddr = '/dev/pts/10' #will need to be replaced with the serial port address. OK to hardcode for now, but eventually your launch file should handle this
#serialPort = serial.Serial(serialPortAddr, 4800, timeout=0.1) #This line opens the port


#initilize to make code work even without serial running
#gpggaRead1 = '$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,TSTR*61'

#Stuff you will do in a loop to read the incoming GPGGA datat:
# if serialPort.in_waiting > 0:
#     # Read available data
#     gpggaRead1 = serialPort.readline().decode('utf-8')
#     print(gpggaRead1) #You can delete this once you confirm it works

#Code you wrote above to handle the GPGGA strings:
# #gpggaRead = ''

######### implemented above in timer_callback
#     def isGPGGAinString(self, inputString):
#         if '$GPGGA' in inputString: #replace 1 == 1 with condition to be checked for inputString
#             #global gpggaRead
#             print('Great success!')
#             #gpggaRead = inputString
#         else:
#             print('GPGGA not found in string')

    #Check back to see if port string is gpgga, TEST when writing code. It worked!
    # stringReadfromPort = 'dfhdhg'
    # isGPGGAinString(stringReadfromPort)
    def parse_vnymr(self, vnymr_read):
        try:
            # Yaw, Pitch, Roll, Magnetic, Acceleration, and Angular Rate Measurements
            # String format:    $VNYMR,Yaw,Pitch,    Roll,    MagX,    MagY,   MagZ,  AccelX,AccelY,AccelZ,    GyroX,GyroY,    GyroZ*CS
            # Example Response: $VNRRG,27,+006.380,+000.023,-001.953,+1.0640,-0.2531,+3.0614,+00.005,+00.344,-09.758,-0.001222,-0.000450,-0.001218*4F
            vnymrSplit = vnymr_read.strip().split(",") #Put code here that will split gpggaRead into its components. This should only take one line.
            print(vnymrSplit)


            # UTC = float(vnymrSplit[1])
            # Latitude = float(vnymrSplit[2])
            # LatitudeDir = str(vnymrSplit[3])
            # Longitude = float(vnymrSplit[4])
            # LongitudeDir = str(vnymrSplit[5])
            # HDOP = float(vnymrSplit[8])
            # Altitude = float(vnymrSplit[9])
            #
            yaw = float(vnymrSplit[1]) # in degrees
            pitch = float(vnymrSplit[2]) # in degrees
            roll = float(vnymrSplit[3]) # in degrees

            #r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
            #[quatx, quaty, quatz, quatw] = r.as_quat()  # returns [x, y, z, w]

            [quatx, quaty, quatz, quatw] = self.convert_to_quaternion(yaw, pitch, roll)

            #mag field in gauss from VectorNav, convert to Tesla for Ros2 MagneticField: gauss/10,000 = tesla
            MagX = (float(vnymrSplit[4]))/10000 # converted to tesla
            MagY = (float(vnymrSplit[5]))/10000 # converted to tesla
            MagZ = (float(vnymrSplit[6]))/10000 # converted to tesla
            AccelX = float(vnymrSplit[7]) # in m/s^2
            AccelY = float(vnymrSplit[8]) # in m/s^2
            AccelZ = float(vnymrSplit[9]) # in m/s^2
            GyroX = float(vnymrSplit[10]) # in rad/s
            GyroY = float(vnymrSplit[11]) # in rad/s
            original_string = vnymrSplit[12]
            new_string = original_string[:-3]
            print(new_string)
            GyroZ = float(new_string) # in rad/s

            # print('UTC '+str(UTC))
            # print('Latitude '+str(Latitude))
            # print(LatitudeDir)
            # print('Longitude '+str(Longitude))
            # print(LongitudeDir)
            # print('HDOP '+str(HDOP))

            # Latitude = self.degMinstoDegDec(Latitude)
            # Longitude = self.degMinstoDegDec(Longitude)
            # LatitudeSigned = self.LatLongSignConvention(Latitude, LatitudeDir)
            # LongitudeSigned = self.LatLongSignConvention(Longitude, LongitudeDir)
            # #Altitude =
            # easting, northing, zone, letter = self.convertToUTM(LatitudeSigned, LongitudeSigned)
            # time_sec, time_nsec = self.UTCtoUTCEpoch(UTC)

            curr_time_ns = time.time_ns()
            curr_time_sec = curr_time_ns // 1_000_000_000 # just gives whole seconds and removes remainder nanosecs
            curr_time_nsec = curr_time_ns % 1_000_000_000 # % finds remainder number of int nanoseconds

            print('startmsg')

            header = Header()
            header.stamp.sec = curr_time_sec
            header.stamp.nanosec = curr_time_nsec
            header.frame_id = "imu1_frame"

            print('mid')

            mf_msg = MagneticField()
            mf_msg.header = header
            mf_msg.magnetic_field.x = MagX
            mf_msg.magnetic_field.y = MagY
            mf_msg.magnetic_field.z = MagZ
            mf_msg.magnetic_field_covariance = [0, 0, 0]

            imu_msg = Imu()
            imu_msg.header = header
            imu_msg.orientation.x = quatx
            imu_msg.orientation.y = quaty
            imu_msg.orientation.z = quatz
            imu_msg.orientation.w = quatw
            imu_msg.orientation_covariance = [0, 0, 0]

            imu_msg.linear_acceleration.x = AccelX
            imu_msg.linear_acceleration.y = AccelY
            imu_msg.linear_acceleration.z = AccelZ
            imu_msg.linear_acceleration_covariance = [0, 0, 0]

            imu_msg.angular_velocity.x = GyroX
            imu_msg.angular_velocity.y = GyroY
            imu_msg.angular_velocity.z = GyroZ
            imu_msg.angular_velocity_covariance = [0, 0, 0]

            msg = Vectornav()
            msg.header = header
            msg.mag_field = mf_msg
            msg.imu = imu_msg
            msg.raw = vnymr_read

            # msg.yaw = Yaw
            # msg.pitch = Pitch
            # msg.roll = Roll
            # msg.magx = MagX
            # msg.magy = MagY
            # msg.magz = MagZ
            # msg.accelx = AccelX
            # msg.accely = AccelY
            # msg.accelz = AccelZ
            # msg.gyrox = GyroX
            # msg.gyroy = GyroY
            # msg.gyroz = GyroZ
            print('endmsg')
            # msg.latitude = LatitudeSigned
            # msg.longitude = LongitudeSigned
            # msg.altitude = Altitude
            # msg.utm_easting = easting
            # msg.utm_northing = northing
            # msg.zone = zone
            # msg.letter = letter
            # msg.hdop = HDOP
            # msg.gpgga_read = gpgga_read.strip()

            return msg
        except Exception as e:
            self.get_logger().error(f'Failed to parse VNYMR string: {e}')
            return None


    def convert_to_quaternion(self, yaw, pitch, roll):
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
        [quatx, quaty, quatz, quatw] = r.as_quat()  # returns [x, y, z, w]
        return quatx, quaty, quatz, quatw

    def degMinstoDegDec(self, LatOrLong):
        deg = int(LatOrLong/100) #Replace 0 with code that gets just the degrees from LatOrLong
        mins = float(LatOrLong)%100 #Replace 0 with code that gets just the minutes from LatOrLong
        degDec = round(mins/60, 15) #Replace 0 with code that converts minutes to decimal degrees
        print(deg+degDec)
        return (deg+degDec)
    #print('lat long in DD.dddd:')
    #Latitude = degMinstoDegDec(Latitude)
    #Longitude = degMinstoDegDec(Longitude)

    def LatLongSignConvention(self, LatOrLong, LatOrLongDir):
        if LatOrLongDir == "W" or LatOrLongDir == "S": #Replace the blank string with a value
            LatOrLong = LatOrLong*(-1) #some code here that applies negative convention
        print(LatOrLong)
        return LatOrLong
    # print('Latitude Signed:')
    # LatitudeSigned = LatLongSignConvetion(Latitude, LatitudeDir)
    # print('Longitude Signed:')
    # LongitudeSigned = LatLongSignConvetion(Longitude, LongitudeDir)

    def convertToUTM(self, LatitudeSigned, LongitudeSigned):
        UTMVals = utm.from_latlon(LatitudeSigned, LongitudeSigned)
        UTMEasting = UTMVals[0] #Again, replace these with values from UTMVals
        UTMNorthing = UTMVals[1]
        UTMZone = UTMVals[2]
        UTMLetter = UTMVals[3]
        print(UTMVals)
        return [UTMEasting, UTMNorthing, UTMZone, UTMLetter]
    # print('UTM:')
    # convertToUTM(LatitudeSigned, LongitudeSigned)

    def UTCtoUTCEpoch(self, UTC):
        hours = int(UTC/10000) #UTC in HHMMSS.SS
        sec = float(UTC)%100
        min = int((UTC-(hours*10000+sec))/100)
        UTCinSecs = 3600*(hours)+60*(min)+(sec) #Replace with code that converts the UTC float in hhmmss.ss to seconds as a float
        TimeSinceEpoch = time.time() #Replace with code to get time since epoch
        gmtime = time.gmtime(None)
        hour = gmtime.tm_hour
        minute = gmtime.tm_min
        seconds = gmtime.tm_sec
        TimeSinceEpochBOD = TimeSinceEpoch - hour*3600 -minute*60 -seconds#Use the time since epoch to get the time since epoch *at the beginning of the day*
        CurrentTime = TimeSinceEpochBOD + UTCinSecs
        CurrentTimeSec = int(CurrentTime) #Replace with code to get total seconds as an integer
        CurrentTimeNsec = int((CurrentTime-CurrentTimeSec)*1000000000)#time.time_ns() - ()#Replace with code to get remaining nanoseconds as an integer (between CurrentTime and CurrentTimeSec )
        print ('UTC',UTC), print('hr  ', hours), print('min ', min), print('s   ', sec)
        print('UTC in seconds:                    ', UTCinSecs)
        print('current time since epoch:           ', CurrentTime)
        print('time since epoch whole seconds:    ', CurrentTimeSec)
        print('time since epoch remaining nano secs:  ', CurrentTimeNsec)
        return [CurrentTimeSec, CurrentTimeNsec]
    ### should i use  time.time_ns() → int
    # Similar to time() but returns time as an integer number of nanoseconds since the epoch.
    # print(current_time_secs)
    # print(time.localtime(None))
    # CurrentTime = UTCtoUTCEpoch(UTC)

    def destroy_node(self):
        if self.serialPort:
            self.serialPort.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    node = VNDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#Stuff you will do when the node is shutdown:
#serialPort.close() #Do not modify
