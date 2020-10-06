"""
all functions needed to setup/customize parameters of "Flir" camera:
    
http://softwareservices.ptgrey.com/Spinnaker/latest/page2.html
https://www.flir.com/support-center/iis/machine-vision/application-note/spinnaker-nodes/
"""

import PySpin

def configure_frame_rate(cam, 
                         frame_rate_enable = False,
                         frame_rate_to_set = 30.0):
    
    # set gain
    nodemap = cam.GetNodeMap()
    node_frame_auto = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
    if not PySpin.IsAvailable(node_frame_auto) or not PySpin.IsWritable(node_frame_auto):
        print('Unable to change frame rate mode (node retrieval). Aborting...')

    node_frame_auto.SetValue(frame_rate_enable)
    print('Frame rate enable is %s...' %frame_rate_enable)
    
    if frame_rate_enable == True:
        node_frame = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_frame) or not PySpin.IsWritable(node_frame):
            print('Unable to get frame rate (node retrieval). Aborting...')
    
        node_frame.SetValue(frame_rate_to_set)
        print('\tframe rate set to {0:.5f}...'.format(node_frame.GetValue()))


def configure_gain_gamma_white_balance(cam,
                                       gain_auto = 'Off',
                                       white_auto = 'Off',
                                       gain_to_set = 4.0, 
                                       gamma_to_set = 1.0):
    
    # set gamma
    cam.Gamma.SetValue(gamma_to_set)
    
    # set gain
    nodemap = cam.GetNodeMap()
    node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
    if not PySpin.IsAvailable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
        print('Unable to change gain mode (node retrieval). Aborting...')

    gain_auto_off = node_gain_auto.GetEntryByName(gain_auto)
    if not PySpin.IsAvailable(gain_auto_off) or not PySpin.IsReadable(gain_auto_off):
        print('Unable to change gain mode (node retrieval). Aborting...')

    node_gain_auto.SetIntValue(gain_auto_off.GetValue())
    print('Automatic gain %s...' %gain_auto)
    
    if gain_auto == 'Off':
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_gain) or not PySpin.IsWritable(node_gain):
            print('Unable to get gain (node retrieval). Aborting...')

        node_gain.SetValue(gain_to_set)
        print('\tGain set to {0:.5f}...'.format(node_gain.GetValue()))
    
    # set white balance
    node_white_auto = PySpin.CEnumerationPtr(nodemap.GetNode("BalanceWhiteAuto"))
    if not PySpin.IsAvailable(node_white_auto) or not PySpin.IsWritable(node_white_auto):
        print('Unable to change white balance mode')
    
    else:
        white_auto_off = node_white_auto.GetEntryByName(white_auto)
        if not PySpin.IsAvailable(white_auto_off) or not PySpin.IsReadable(white_auto_off):
            print('Unable to change white balance mode')
    
        node_gain_auto.SetIntValue(white_auto_off.GetValue())
        print('Automatic white balance %s...' %white_auto)


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

def configure_trigger(cam, 
                      trigger_mode = 'On',
                      CHOSEN_TRIGGER = 2,
                      line_num = 3,
                      trigger_edge = 'RisingEdge'):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.
    """

    print('*** CONFIGURING TRIGGER ***\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE and trigger_mode == 'On':
        print('Software trigger chosen ...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE and trigger_mode == 'On':
        print('Hardware trigger chose ...')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')
        
        if trigger_mode == 'On':
            # Select trigger source
            # The trigger source must be set to hardware or software while trigger
            # mode is off.
            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
            if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print('Unable to get trigger source (node retrieval). Aborting...')
    
            if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
                node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
                if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                        node_trigger_source_software):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
    
            elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
                node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line%d' %line_num)
                if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                        node_trigger_source_hardware):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
                
            # Select hardware trigger edge
            if CHOSEN_TRIGGER == TriggerType.HARDWARE:
                node_trigger_edge = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerActivation'))
                if not PySpin.IsAvailable(node_trigger_edge) or not PySpin.IsWritable(node_trigger_edge):
                    print('Unable to get trigger edge (node retrieval). Aborting...')
    
                node_trigger_edge_hardware = node_trigger_edge.GetEntryByName(trigger_edge)
                if not PySpin.IsAvailable(node_trigger_edge_hardware) or not PySpin.IsReadable(
                        node_trigger_edge_hardware):
                    print('Unable to set trigger edge (enum entry retrieval). Aborting...')
                node_trigger_edge.SetIntValue(node_trigger_edge_hardware.GetValue())
                
            # Turn trigger mode on
            # Once the appropriate trigger source has been set, turn trigger mode
            # on in order to retrieve images using the trigger.
            node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
            if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
                print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
    
            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
            print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)


def configure_custom_image_settings(cam, 
                                    pixel_format = "mono8"):
    """
    Configures a number of settings on the camera including offsets X and Y,
    width, height, and pixel format. These settings must be applied before
    BeginAcquisition() is called; otherwise, those nodes would be read only.
    Also, it is important to note that settings are applied immediately.
    This means if you plan to reduce the width and move the x offset accordingly,
    you need to apply such changes in the appropriate order.

    """
    print('\n*** CONFIGURING CUSTOM IMAGE SETTINGS ***\n')

    try:

        if cam.PixelFormat.GetAccessMode() == PySpin.RW:
            if pixel_format == "mono8":
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            elif pixel_format == "bgr8":
                cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
            print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())

        else:
            print('Pixel format not available...')
            
        if cam.OffsetX.GetAccessMode() == PySpin.RW:
            cam.OffsetX.SetValue(cam.OffsetX.GetMin())
            print('Offset X set to %d...' % cam.OffsetX.GetValue())

        else:
            print('Offset X not available...')

        if cam.OffsetY.GetAccessMode() == PySpin.RW:
            cam.OffsetY.SetValue(cam.OffsetY.GetMin())
            print('Offset Y set to %d...' % cam.OffsetY.GetValue())

        else:
            print('Offset Y not available...')

        if cam.Width.GetAccessMode() == PySpin.RW and cam.Width.GetInc() != 0 and cam.Width.GetMax != 0:
            cam.Width.SetValue(cam.Width.GetMax())
            print('Width set to %i...' % cam.Width.GetValue())

        else:
            print('Width not available...')

        if cam.Height.GetAccessMode() == PySpin.RW and cam.Height.GetInc() != 0 and cam.Height.GetMax != 0:
            cam.Height.SetValue(cam.Height.GetMax())
            print('Height set to %i...' % cam.Height.GetValue())

        else:
            print('Height not available...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)



def configure_exposure(cam,
                       exposure_auto = 'Off',
                       exposure_time_to_set = 6700):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.
    """

    print('*** CONFIGURING EXPOSURE ***\n')

    try:

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to change automatic exposure. Aborting...')

        if exposure_auto == 'Off':
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        else:
            exposure_auto = 'Continuous'
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        
        print('Automatic exposure %s...' %exposure_auto)

        if exposure_auto == 'Off':
            if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                print('Unable to set exposure time. Aborting...')
    
            # Ensure desired exposure time does not exceed the maximum
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Shutter time set to %s us...\n' % exposure_time_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)


def configure_acquisition(cam, nodemap, nodemap_tldevice):
    """
    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    """

    try:

        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')


        print('Camera settings configured...')

        #  Retrieve device serial number
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)



def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)




def configure_single_camera(cam, cam_params):

    try:
        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()
                       
        configure_exposure(cam, exposure_auto = cam_params.exposure_auto,
                           exposure_time_to_set = cam_params.exposure_time_to_set)
        configure_custom_image_settings(cam, 
                                        pixel_format = cam_params.pixel_format)
        
        configure_gain_gamma_white_balance(cam, gain_auto = cam_params.gain_auto,
                                           white_auto = cam_params.white_auto,
                                           gain_to_set = cam_params.gain_to_set,
                                           gamma_to_set = cam_params.gamma_to_set)
        configure_trigger(cam, trigger_mode = cam_params.trigger_mode,
                          CHOSEN_TRIGGER = cam_params.CHOSEN_TRIGGER,
                          line_num = cam_params.line_num,
                          trigger_edge = cam_params.trigger_edge)
        
        configure_frame_rate(cam,
                             frame_rate_enable = cam_params.frame_rate_enable,
                             frame_rate_to_set = cam_params.frame_rate_to_set)
        
        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        configure_acquisition(cam, nodemap, nodemap_tldevice)
        
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    

def configure_camera(cam_params, cam_id = 0):

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')

    # Run camera with cam_id
    i, cam = list(enumerate(cam_list))[cam_id]

    print('Running camera %d...' % i)

    configure_single_camera(cam, cam_params)
    
    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()