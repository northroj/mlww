# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:25:13 2023

@author: JorNo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

trials = 13000
numcells = 10
totaldata = np.zeros((trials, 10*numcells+1))


for trial in range(trials):
    
    data = np.zeros((1,numcells,2,5))
    
    # data[set,mat,group,xs]
    # xs:  0 = t, 1 = igs, 2 = ds, 3 = nsf, 4 = chi
    data[0,:,0,1] = np.random.uniform(0.01,0.5,numcells)
    data[0,:,1,1] = np.random.uniform(0.01,1.0,numcells)
    data[0,:,0,2] = np.random.uniform(0.01,0.1,numcells)
    data[0,:,0,3] = np.random.uniform(0.01,0.05,numcells)
    data[0,:,1,3] = np.random.uniform(0.01,0.7,numcells)
    data[0,:,0,4] = np.random.randint(0,2,numcells).astype(float)
    capture_fast = np.random.uniform(0.01,0.02,numcells)
    capture_thermal = np.random.uniform(0.1,0.5,numcells)
    data[0,:,0,0] = data[0,:,0,1] + data[0,:,0,2] + capture_fast
    data[0,:,1,0] = data[0,:,1,1] + data[0,:,1,2] + capture_thermal
    
    dataset = 0
    
    
    #initialize first fission source guess and k
    fissource = np.ones((numcells))/numcells
    keff = 1
    
    convergePI = 0
    convergek = 0
    errSI = 1e-6
    errk = 1e-5
    errPI = 1e-5
    
    
    dx = 20/numcells
    numgroups = 2
    CAfluxold = np.ones((numgroups,numcells))
    CAfluxstore = np.zeros((numgroups,numcells))
    CEcurrent = np.zeros((numgroups,numcells+1))
    CEflux = np.zeros((numgroups,numcells+1))
    
    materials = np.arange(0,numcells,1)

    Sord = 4
    halfS = int(Sord/2)
    
    if Sord == 4:
        quads = np.array([-0.861136311594, -0.339981043585, 0.339981043585, 0.861136311594])
        weights = np.array([0.347854845149, 0.652145154858, 0.652145154858, 0.347854845149])
    else:
        quads = np.array([-0.577350269190, 0.577350269190])
        weights = np.array([1,1])
    
    #negative then positive angles
    BCangflux = np.ones((numgroups,Sord))*0
    
    
    maxPIit = 50
    maxSIit = 100
    PIit = 0
    while (convergePI == 0 or convergek == 0):
        
        
        
        for group in range(numgroups):
            
            Sg = np.zeros((numcells))
            for newsourceit in range(numcells):
                #this methodology only works for 2 groups
                if group == 0:
                    downscatter = 0
                else:
                    downscatter = data[dataset,materials[newsourceit],0,2] * CAfluxold[0,newsourceit]
                upscatter = 0
                #inscatter = data[dataset,materials[newsourceit],group,1]*CAfluxold[group,newsourceit]
                Sg[newsourceit] = data[dataset,materials[newsourceit],group,4]*fissource[newsourceit]/keff + downscatter + upscatter # + inscatter
            
            
            
            convergeSI = 0
            SIit = 0
            while convergeSI == 0:
                CAfluxstore = CAfluxold * 1
                CEfluxnew = np.zeros(numcells+1)
                CEcurrentnew = np.zeros(numcells+1)
                
                Qg = np.zeros(numcells)
                for newqit in range(numcells):
                    Qg[newqit] = (Sg[newqit] + data[dataset,materials[newqit],group,1] * CAfluxold[group,newqit])*0.5
                CAfluxnewgroup = np.zeros((numcells))
                
                
                #Positive angles
                for angle in range(halfS):
                                    
                    currentquad = quads[angle + halfS]
                    currentweight = weights[angle + halfS]
                    CAangflux = np.zeros((numcells))
                    exitangflux = np.zeros((numcells))
                    for cell in range(numcells):
                        
                        if cell == 0:
                            exitangflux[cell] = (dx*Qg[cell]+BCangflux[group,angle+halfS]*(currentquad-data[dataset,materials[cell],group,0]*dx*0.5))/(currentquad+data[dataset,materials[cell],group,0]*dx*0.5)
                            CAangflux[cell] = 0.5*(exitangflux[cell] + BCangflux[group,angle+halfS])
                        else:
                            exitangflux[cell] = (dx*Qg[cell]+exitangflux[cell-1]*(currentquad-data[dataset,materials[cell],group,0]*dx*0.5))/(currentquad+data[dataset,materials[cell],group,0]*dx*0.5)
                            CAangflux[cell] = 0.5*(exitangflux[cell] + exitangflux[cell-1])
                        CAfluxnewgroup[cell] += currentweight * CAangflux[cell]
                        if (cell > 0)  and (cell < numcells):
                            CEfluxnew[cell] += currentweight * exitangflux[cell]
                            CEcurrentnew[cell] += currentweight * exitangflux[cell]
            
                    BCangflux[group,halfS-angle-1] = exitangflux[numcells-1]
                    CEfluxnew[numcells] += currentweight * BCangflux[group,halfS-angle-1]
                    
                
                #Negative angles
                for angle in range(halfS):
                    
                    currentquad = abs(quads[angle])
                    currentweight = weights[angle]
                    CAangflux = np.zeros((numcells))
                    exitangflux = np.zeros((numcells))
                    
                    for cell in range(numcells-1, -1, -1):
                        
                        if cell == numcells-1:
                            exitangflux[cell] = (dx*Qg[cell]+BCangflux[group,angle]*(currentquad-data[dataset,materials[cell],group,0]*dx*0.5))/(currentquad+data[dataset,materials[cell],group,0]*dx*0.5)
                            CAangflux[cell] = 0.5*(exitangflux[cell] + BCangflux[group,angle])
                        else:
                            exitangflux[cell] = (dx*Qg[cell]+exitangflux[cell+1]*(currentquad-data[dataset,materials[cell],group,0]*dx*0.5))/(currentquad+data[dataset,materials[cell],group,0]*dx*0.5)
                            CAangflux[cell] = 0.5*(exitangflux[cell] + exitangflux[cell+1])
                        #name7 = input('')
                        CAfluxnewgroup[cell] += currentweight * CAangflux[cell]
                        if (cell > 0)  and (cell < numcells):
                            CEfluxnew[cell] += currentweight * exitangflux[cell]
                            CEcurrentnew[cell] -= currentweight * exitangflux[cell]
                        
                    BCangflux[group,Sord-angle-1] = exitangflux[0]
                    CEfluxnew[0] += currentweight * BCangflux[group,Sord-angle-1]
                    
                    
                CAfluxold[group,:] = CAfluxnewgroup * 1
                CEflux[group,:] = CEfluxnew[:]*1
                CEcurrent[group,:] = CEcurrentnew*1
                
                
                maxdif = 0
                for sicheck in range(numcells):
                    if np.abs((CAfluxold[group,sicheck] - CAfluxstore[group,sicheck])/CAfluxold[group,sicheck]) > maxdif:
                        maxdif = np.abs((CAfluxold[group,sicheck] - CAfluxstore[group,sicheck])/CAfluxold[group,sicheck])
                if maxdif < errSI:
                    convergeSI = 1
                SIit +=1
                if SIit > maxSIit-1:
                    print("max SI iterations reached on PI cycle: ", PIit+1, ", group: ", group, "\n")
                    break
            #name3 = input('group it')
        fissourceold = fissource*1
        for newfis in range(numcells):
            fissource[newfis] = data[dataset,materials[newfis],1,3]*CAfluxold[1,newfis]
        #print(np.sum(fissource))
        kstore = keff*1
        keff = keff * np.sum(fissource) / np.sum(fissourceold)  #implied *dx in each cell
        
        if np.isnan(keff) == True or keff < 0.00000001:
            keff = 1000.0
            break
        
        kdif = np.abs((keff-kstore)/keff)
        
        maxfisdif = 0
        for picheck in range(numcells):
            if np.abs(fissourceold[picheck] - fissource[picheck]) > maxfisdif:
                maxfisdif = np.abs(fissourceold[picheck] - fissource[picheck])
        if (maxfisdif < errPI):
            convergePI = 1
            
        if (kdif < errk):
            convergek = 1
        
        print('cycle: ', PIit+1, ', keff: ', keff)
        
        PIit +=1
        #name = input('pi it')
        if PIit > maxPIit-1:
            print("max PI iterations reached\n")
            break
    
    print('Outer iterations: ', PIit, '\nkeff = ', keff)
    
    CEflux[0,0] = BCangflux[0,2]+BCangflux[0,3]
    CEflux[0,numcells] = BCangflux[0,0]+BCangflux[0,1]
    CEflux[1,0] = BCangflux[1,2]+BCangflux[1,3]
    CEflux[1,numcells] = BCangflux[1,0]+BCangflux[1,1]
        
    
    
    CAx = np.linspace(0+dx/2, 20-dx/2, numcells)
    CEx = np.linspace(0,20,numcells+1)
    plt.figure(1)
    plt.title('Cell averaged fast flux')
    plt.ylim([np.min(CAfluxold[0,:])/1.1,np.max(CAfluxold[0,:])*1.1])
    plt.plot(CAx,CAfluxold[0,:])
    plt.show()
    plt.figure(2)
    plt.title('Cell averaged thermal flux')
    plt.ylim([np.min(CAfluxold[1,:])/1.1,np.max(CAfluxold[1,:])*1.1])
    plt.plot(CAx,CAfluxold[1,:])
    plt.show()
    
    data1 = data.flatten(order='f')
    data1 = np.concatenate((data1[:5*numcells],data1[6*numcells:9*numcells]))
    data1[:1*numcells] = capture_fast
    data1[1*numcells:2*numcells] = capture_thermal
    
    # 1 capture f, 2 capture t, 3 igs f, 4 igs t, 5 ds f
    # 6 nsf f, 7 nsf t, 8 chi f, 9 flux f, 10 flux t, 11 keff
    totaldata[trial,:8*numcells] = data1
    totaldata[trial,8*numcells:9*numcells] = CAfluxold[0,:]
    totaldata[trial,9*numcells:10*numcells] = CAfluxold[1,:]
    totaldata[trial,10*numcells] = keff
    
    
DF1 = pd.DataFrame(totaldata)
DF1.to_csv("testdata1.csv")

        