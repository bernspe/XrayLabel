import cv2
import numpy as np
import json
import os
import sys
import scipy.interpolate as ip
from scipy.signal import argrelextrema
from skimage.draw import line
import pandas as pd
import pickle

#get config data
config_dict = json.load(open('scol_xray_exp1.json'))
resizedxraydir=config_dict['resizedxraydir']
xraydf=config_dict['xraydf']
picparquet=config_dict['imgsource']

#organize protocol dir
xrayprotocoldir=config_dict['xray_protocol']
if not os.path.exists(xrayprotocoldir):
    os.makedirs(xrayprotocoldir)

#get label and image data
try:
    df=pd.read_pickle(xraydf)
except:
    print('Missing label file')
    sys.exit()

try:
    pic_df=pd.read_parquet(picparquet)
except:
    print('Missing source file')
    sys.exit()

#global vars
vertebrae=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5','S1']
angles_vertebrae=['aT1','aT2','aT3','aT4','aT5','aT6','aT7','aT8','aT9','aT10','aT11','aT12','aL1','aL2','aL3','aL4','aL5','aS1']
showall= False
validation_mode=True
validate=False    
offset=30
threshold=100
coords=[]
tempcoords=[]
mancoords=[]
ix,iy=0,0
img=np.zeros((500,500),dtype=np.uint8)
edg=np.zeros((500,500),dtype=np.uint8)
tempimg=np.zeros((500,500),dtype=np.uint8)
partimg=np.zeros((50,50),dtype=np.uint8)

# getting the angle of 3 points (corner = p1), points = [x,y]
def get_angle(p0, p1=np.array([0,0]), p2=None):
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def adjustuv(x):
    #print('Upper vertebra adjusted to: %s'%(vertebrae[x]))
    pass

def adjustlv(x):
    #print('Lower vertebra adjusted to: %s'%(vertebrae[x]))
    pass

def draw_circle(event,x,y,flags,param):
    global img, partimg, coords, ix,iy,validate,tempcoords
    if event == cv2.EVENT_MOUSEMOVE:
        ix=x
        iy=y
        h,w,_=img.shape
        if (ix>offset) & (ix<w-offset) & (iy>offset) & (iy<h-offset):
            img_part=img[iy-offset:iy+offset,ix-offset:ix+offset]
            partimg=process_part(img_part,ix,iy)
                    
    if event == cv2.EVENT_LBUTTONDOWN:     
        if validate:
            cv2.circle(img,(x,y),3,(255,0,0),1)
            tempcoords.append((x,y))
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
            coords.append((x,y))
        
def process_part(imgpart,ix,iy):
    global tempcoords
    blurred=cv2.GaussianBlur(imgpart[:,:,2],(5,5),0)
    #OTSU Threshold
    ret,thresh=cv2.threshold(blurred,0,threshold,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w,_=imgpart.shape
    resfactor=3
    res = cv2.resize(thresh,(resfactor*w, resfactor*h), interpolation = cv2.INTER_CUBIC)
    imgpart2=cv2.resize(imgpart,(resfactor*w, resfactor*h), interpolation = cv2.INTER_CUBIC)
    hr,hw=res.shape
    m=np.where(res[hr//2,:]>30)[0]
    if len(m)>10:
        cv2.circle(imgpart2,(m[0],hr//2),2,(0,128,255),1)
        cv2.circle(imgpart2,(m[-1],hr//2),2,(0,128,255),1)
        hl=(m[-1]-m[0])//2+m[0]
        cv2.line(imgpart2,(hl,hr//2-10),(hl,hr//2+10),(255,128,128),3)
    cv2.circle(imgpart2,(hw//2,hr//2),8,(0,255,0),2)
    if (validate & ((len(tempcoords) % 2) > 0)):
            x1=(tempcoords[-1][0]-ix+w//2)*resfactor
            y1=(tempcoords[-1][1]-iy+h//2)*resfactor
            cv2.line(imgpart2,(x1,y1),(hw//2,hr//2),(255,128,128),2)
    return imgpart2


#***** MAIN *****
curpos=-1
for row in df.itertuples():
    k=0
    p=row[0]
    dataMode=df.loc[p,'DataMode']
    if type(dataMode) == str:
        if (('fixed' in dataMode) and (not showall)):  # bereits fixierte Bilder überspringen
            curpos+=1
            continue
    #fn=resizedxraydir+p
    #img=cv2.imread(fn)
    img=np.array(pic_df[p].iloc[0], dtype=np.uint8).reshape((500,500,3))
    i2=img.copy()   # backup image
    
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    imgrgb=imgrgb.astype(np.float32)
    

    h,w,_=img.shape
    
    tempimg=img.copy()
    
    dtype=[('x',int),('y',int)]

    cv2.namedWindow('image %s' %p)
    cv2.moveWindow('image %s' %p,100,100)
    cv2.createTrackbar("Upper Vertebra", 'image %s' %p, 0, len(vertebrae)-1, adjustuv)
    cv2.createTrackbar("Lower Vertebra" , 'image %s' %p, len(vertebrae)-2, len(vertebrae)-1, adjustlv)
    
    cv2.imshow('Section',partimg)
    cv2.moveWindow('Section',600,300)
    
    cv2.setMouseCallback('image %s' %p,draw_circle)
    
    #check for existing data
    uv_v=df.loc[p,'upperVertebra']
    lv_v=df.loc[p,'lowerVertebra']

    if (uv_v in vertebrae) & (lv_v in vertebrae):
        cv2.setTrackbarPos("Upper Vertebra", 'image %s' %p, vertebrae.index(uv_v)) 
        cv2.setTrackbarPos("Lower Vertebra", 'image %s' %p, vertebrae.index(lv_v)) 
        
        for pt in df.loc[p,uv_v:lv_v]:
            x=pt[0]
            y=pt[1]
            cv2.circle(img,(x,y),5,(0,0,255),1)
            coords.append((x,y))
            
    curpos+=1
    print('Working %s, which is %i of %i'%(p,curpos,len(df.index)))
    
    while True:
        cv2.imshow('image %s' %p,img)
        cv2.imshow('Section',partimg)
        
        uv=cv2.getTrackbarPos("Upper Vertebra",'image %s' %p)
        lv=cv2.getTrackbarPos("Lower Vertebra",'image %s' %p)
        
        img[0:50,0:100]=np.zeros((50,100,3),dtype=np.uint8)
        cv2.putText(img, 'Slider UV: %s'%vertebrae[uv] , (3,13), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 128), lineType=cv2.LINE_AA)
        cv2.putText(img, 'Slider LV: %s'%vertebrae[lv] , (3,33), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 128), lineType=cv2.LINE_AA)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k==ord('+'):
            threshold+=10
            
        if k==ord('-'):
            threshold-=10
            
        if ((k>=ord('0')) & (k<ord('6'))):
            factor=k-48
            offset=factor*10
                
        if k== ord('d'):  #discard lines
            img=i2.copy()
            tempimg=img.copy()
            coords=[]
            tempcoords=[]
            
        if k==ord('r'): # remove file and data
            df.drop([p],inplace=True)
            #os.remove(fn)
            break

        if k==ord('l'): # make lines
            tempimg=img.copy()
            #sort the coords
            vline=np.array(coords, dtype=dtype)
            vline=np.sort(vline, order='y')
            c_arr=vline.view((int,2))
            # label vertebrae according to slider
            uv=cv2.getTrackbarPos("Upper Vertebra",'image %s' %p)
            lv=cv2.getTrackbarPos("Lower Vertebra",'image %s' %p)
            if c_arr[:,0].shape[0] < (lv-uv):
                print('Lower Vertebra not specified. Only %i entries, expected %i'%(c_arr[:,0].shape[0],lv-uv))
                continue
            else:
                cv2.putText(img, vertebrae[uv] , (c_arr[0,0]-25,c_arr[0,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), lineType=cv2.LINE_AA)
                cv2.putText(img, vertebrae[lv] , (c_arr[-1,0]-25,c_arr[-1,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), lineType=cv2.LINE_AA)
                df.at[p,'upperVertebra']=vertebrae[uv]
                df.at[p,'lowerVertebra']=vertebrae[lv]
            #interpolate mid line
            f1=ip.interp1d(c_arr[:,1], c_arr[:,0], kind='cubic')
            interp_line=[(int(f1(y)),y) for y in range(min(c_arr[:,1]), max(c_arr[:,1])) ]
            for pt in interp_line:
                cv2.circle(img,(pt[0],pt[1]),1,(255,255,0),1)
            for i in range(len(c_arr)):
                df.at[p,vertebrae[uv+i]]=c_arr[i]
            #calculate angles of perpendicular lines
            a=np.zeros(len(c_arr))
            for i in range(1,len(c_arr)-1):
                x1=c_arr[i-1,0]
                x2=c_arr[i+1,0]
                y1=c_arr[i-1,1]
                y2=c_arr[i+1,1]
                midX=(x1+x2)//2
                midY=(y1+y2)//2
                x3=midX-y2+y1 #perpendicular point
                y3=midY + x2-x1
                cv2.line(img,(midX,midY),(x3,y3),(255,128,128),1)
                vangle=get_angle([midX,y3], [x3,y3], [midX,midY])
                a[i]=vangle
                df.at[p,angles_vertebrae[uv+i]]=vangle
                cv2.putText(img, '%.1f'%vangle, (midX,midY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 128), lineType=cv2.LINE_AA)
            
            #calculate COBB angle
            pa=argrelextrema(a, np.greater,order = 2, mode='wrap')[0]
            na=argrelextrema(a, np.less, order = 2, mode='wrap')[0]
            ai=np.sort(np.hstack([pa,na]))
            ai=np.delete(ai,np.where(np.diff(ai)==1)[0])
            cobbs=[]
            del_i=[]
            for i in range(ai.shape[0]-1):
                a_sum=abs(a[ai[i]])+abs(a[ai[i+1]])
                if a_sum>10:
                    cobbs.append(int(a_sum))
                else:
                    del_i.append(i)
            if len(del_i)>0:
                ai=np.delete(ai,del_i)
            #label Neutral-Vertebrae and COBB angle
            df.at[p,'COBB_vertebrae']=[]
            df.at[p,'COBB_angles']=[]
            for i in range(len(ai)-1):
                a1=ai[i]
                a2=ai[i+1]
                cobb=cobbs[i]
                ypos=c_arr[a1][1]+(c_arr[a2][1]-c_arr[a1][1])//2
                xpos=int(c_arr[a2][0]+c_arr[a1][0])//2 + 30
                cv2.putText(img, 'COBB %s - %s: %i'%(vertebrae[uv+a1], vertebrae[uv+a2], cobb), (xpos,ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 128), lineType=cv2.LINE_AA)    
                df.at[p,'COBB_vertebrae'].append((vertebrae[uv+a1],vertebrae[uv+a2]))
                df.at[p,'COBB_angles'].append(cobb)
            
            cv2.imwrite(xrayprotocoldir+'detected_'+p+'.jpg',img)
            
        if k==ord('v'):
            validate=~validate
            if validation_mode & validate:
                img=tempimg.copy()
            if validation_mode & (not validate):
                #group the coords into lines
                tc=np.array(tempcoords, dtype=dtype)
                tc=np.sort(tc, order='y').view((int,2)).reshape((tc.shape[0]//2,4))
                #check if there is an even number of measurepoints
                if (((tc.shape[1]) % 2) == 0):
                    
                    df.at[p,'val_COBB_vertebrae']=[]
                    df.at[p,'val_COBB_angles']=[]
                    #extract y values of vertebrae
                    df_v_y=df.loc[p,'T1':'S1'].apply(lambda x: int(pd.to_numeric(x[1])) if (type(x)!=float) else x)
                    for i in range(tc.shape[0]):
                        lpt=tc[i,:]
                        cv2.line(img,(lpt[0],lpt[1]),(lpt[2],lpt[3]),(255,128,128),1)
                        if i<(tc.shape[0]-1):
                            lpt2=tc[i+1,:]
                            #part-angles are summed up to COBB angle
                            vangle1=abs(get_angle([lpt[0],lpt[1]], [lpt[2],lpt[3]], [lpt[0],lpt[3]]))
                            vangle2=abs(get_angle([lpt2[0],lpt2[1]], [lpt2[2],lpt2[3]], [lpt2[0],lpt2[3]]))
                            if vangle1>125:
                                vangle1=180-vangle1
                            if vangle2>125:
                                vangle2=180-vangle2    
                            sangle=int(vangle1+vangle2)
                            #find the vertebra closest to y value
                            upper_v=df_v_y.sub((lpt[1]+lpt[3])//2).abs().idxmin()
                            lower_v=df_v_y.sub((lpt2[1]+lpt2[3])//2).abs().idxmin()
                            cv2.putText(img, 'COBB %s - %s: %i'%(upper_v, lower_v, sangle), (lpt[0]+50,lpt[1] + ((lpt2[1]-lpt[1])//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 128), lineType=cv2.LINE_AA)
                            df.at[p,'val_COBB_vertebrae'].append((upper_v,lower_v))
                            df.at[p,'val_COBB_angles'].append(sangle)
                            
                            
                else:
                    print("Angle points are not depicted correctly..")
                cv2.imwrite(xrayprotocoldir+'validate_'+p+'.jpg',img)
                tempcoords=[]
                        
        
        if k==ord('s'): # save and seal this data row
            df.at[p, 'DataMode']='fixed'
            df.to_pickle(xraydf)
        if (k==27) | (k==ord('n')) | (k==ord('r')): # break criteria
            break

    #Next Image
    if k==ord('n'):
        coords=[]
        tempcoords=[]
        validate=False
        
        #cv2.imwrite(xraycroppeddir+p,i2)
    elif k==ord('r'):
        coords=[]
        
    # Task Termination
    cv2.destroyAllWindows()
    df.to_pickle(xraydf)
    
    if k==27:
        break

