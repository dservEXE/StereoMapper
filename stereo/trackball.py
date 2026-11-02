"""Camera-space arcball rotation with no Euler-angle singularity."""
import numpy as np


def axis_rotation(axis,angle):
    vector=np.eye(3)[axis]
    x,y,z=vector
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


def arcball_rotation(first,second,width,height):
    def sphere(point):
        p=(np.asarray(point,np.float64)-[width/2,height/2])*2/max(1,min(width,height))
        norm=float(np.dot(p,p))
        if norm>1:return np.array([* (p/np.sqrt(norm)),0.])
        return np.array([*p,np.sqrt(1-norm)])
    a,b=sphere(first),sphere(second)
    vector=np.cross(a,b)
    cosine=np.clip(np.dot(a,b),-1,1)
    sine=np.linalg.norm(vector)
    if sine<1e-10:
        if cosine>0:return np.eye(3)
        vector=np.cross(a,np.eye(3)[np.argmin(np.abs(a))])
        vector/=np.linalg.norm(vector)
        return 2*np.outer(vector,vector)-np.eye(3)
    x,y,z=vector/sine
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+sine*cross+(1-cosine)*(cross@cross)
