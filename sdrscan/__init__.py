import matplotlib.pyplot as plt
import rtlsdr
import numpy as np
import time
import pandas as pd
from datetime import datetime
from scipy import signal
import os
from sklearn.metrics import mean_squared_error


def hacer_potencia(psd_max):
    potencia=10*np.log10(psd_max)
    return potencia


def setup(f_min, f_max,veces):
    #Frecuency range and step
   
    rate_best = 2.4e6
    df = rate_best


    # Set up the scan
    freqs = np.arange(f_min + df/2.,f_max,df)
    nfreq = freqs.shape[0]  
    npsd_res = 1024
    npsd_avg = 256
    nsamp = npsd_res*npsd_avg
    nfreq_spec = nfreq*npsd_res 
    samples = np.zeros([nsamp,nfreq],dtype='complex128') 

    #Setting the data lists 
    psd_array = np.zeros([npsd_res,nfreq])
    freq_array = np.zeros([npsd_res,nfreq])
    #time_array = np.zeros([npsd_res,nfreq],dtype='datetime64[s]')
    relative_power_array = np.zeros([npsd_res,nfreq])


    #Configuracion de dataframes para el MAXHOLD
    len=freq_array.shape[0]
    
    psd_total=np.empty([len*2,veces])

    return rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total


def readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces):
    #Initializing SDR
    sdr = rtlsdr.RtlSdr()
    sdr.sample_rate = rate_best
    sdr.gain = 0
    samp_rate = sdr.sample_rate 
    for k in range(veces):
        for i,freq in enumerate(freqs):
                sdr.center_freq = freq
                samples[:,i] = sdr.read_samples(nsamp)
    for i,freq in enumerate(freqs):   
        fc_mhz = freq/1e6
        bw_mhz = sdr.sample_rate/1e6  
        psd_array[:,i],freq_array[:,i] = plt.psd(samples[:,i], NFFT=npsd_res, Fs=bw_mhz, Fc=fc_mhz)
    #print(psd_array)
    freq_series=np.concatenate(freq_array)
    psd_series=np.concatenate(psd_array)
    #print(k)
    #psd=pd.DataFrame(psd_series)
    psd_total=np.insert(psd_total,k,psd_series,axis=1)

    sdr.close()

    psd_total=pd.DataFrame(psd_total)
    psd_new=psd_total.loc[:,0:veces-1]
    psd_max=psd_new.max(axis=1)

    max_hold=psd_max.apply(hacer_potencia)
    data_array = np.stack((freq_series, max_hold), axis=1)
    df=pd.DataFrame(data_array,columns=['Frecuencia','Potencia'])
    data= df.sort_values('Frecuencia',ascending=True)
    
    return data

def canal_filter(data,f_min_canal,f_max_canal):
    data_canal=data[(data['Frecuencia']>=f_min_canal) & (data['Frecuencia']<=f_max_canal)]
    data_canal=data_canal.reset_index(drop=True)
    return data_canal

def minima_senal_detectable(num):
    return num


def detection_limit(n,umbral,constante):
    if n <= umbral: 
        return constante
    else:
        return n


def comparacion(senal_referencia,senal_comparacion):


    corr=senal_referencia.corr(senal_comparacion)
    corr_validation=np.isnan(corr)
    if corr_validation==True:
        corr=0.2
    rmse=mean_squared_error(senal_referencia,senal_comparacion,squared=True)

    if rmse > 10:
        rmse_list=[]
        for i in range(50):
            rmse=mean_squared_error(senal_referencia,senal_comparacion,squared=True)
            rmse_list.append(rmse)
        rmse=min(rmse_list)
    return corr,rmse


def minimun_signal_detectable(dict,data):
    for key in dict:
        values=dict[key]
        condicicon=values[2]
        if condicicon=='libre':
            data_canal=canal_filter(data,values[0],values[1])
            if data_canal['Potencia'].max() < -25:
                umbral = data_canal['Potencia'].max()
                senal_referencia=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))
                
    print('El umbral es: '+ str(umbral)+' dBm'+' del '+str(key))
    return umbral, senal_referencia   

    
def signal_coherence(senal_referencia,senal_comparacion):
    f, Cxy = signal.coherence(senal_referencia,senal_comparacion)
    return Cxy


def run(data,f_min_canal,f_max_canal,umbral,senal_referencia):
    data_canal=canal_filter(data,f_min_canal,f_max_canal)
    #umbral=data_canal['Potencia'].max()
    #senal_referencia=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))
    senal_comparacion=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))

    if senal_comparacion.shape[0] != senal_referencia.shape[0]:
        senal_comparacion=senal_comparacion[0:senal_referencia.shape[0]]
    
    corr,rmse = comparacion(senal_referencia,senal_comparacion)     #compararmos la senal con la misma solo para probar 
    #coherencia = signal_coherence(senal_referencia,senal_comparacion)
    return corr, data_canal,rmse

def procesamiento(f_min,f_max,canales):
    veces=50
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            corr,data_canal,rmse=run(data,f_min_canal,f_max_canal,umbral,senal_referencia)
            print('El rmse es: '+ str(rmse))
            print('La correlacion es ' + str(corr))
            if corr < 0.5 and rmse > 10 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -20 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    print(parasita)    #necesitas devolver esto en la funcion final
                    print(maxim)
                    #return parasita , 1
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                #return 0