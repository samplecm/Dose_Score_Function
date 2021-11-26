import os 
import pydicom 

path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Research/20211110_Caleb_SGFX/SGFX0024/RD.1.2.246.352.71.7.811746149197.10351637.20141113111248.dcm"


data = pydicom.dcmread(path)


print(data)
print(data[0x0008,0x0060].value)

# roiSequence = data.data_element("StructureSetROISequence")
# for element in roiSequence:
#     print(element.get("ROIName").lower())