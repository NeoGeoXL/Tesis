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


def primera_iteracion():

    f_min = 88e6
    f_max = 92e6
    veces=50
    
    canales ={
        'canal 1': [88.0000,88.19000,'libre'],
        'canal 2': [88.2000,88.39000,'libre'],
        'canal 3': [88.4000,88.59000,'usado'],
        'canal 4': [88.6000,88.79000,'libre'],
        'canal 5': [88.8000,88.99000,'libre'],
        'canal 6': [89.0000,89.19000,'libre'],
        'canal 7': [89.2000,89.39000,'libre'],
        'canal 8': [89.4000,89.59000,'libre'],
        'canal 9': [89.6000,89.79000,'usado'],
        'canal 10': [89.8000,89.99000,'libre'],
        'canal 11': [90.0000,90.19000,'usado'],
        'canal 12': [90.2000,90.39000,'libre'],
        'canal 13': [90.4000,90.59000,'usado'],
        'canal 14': [90.6000,90.79000,'libre'],
        'canal 15': [90.8000,90.99000,'usado'],
        'canal 16': [91.0000,91.19000,'libre'],
        'canal 17': [91.2000,91.39000,'usado'],
        'canal 18': [91.4000,91.59000,'libre'],
        'canal 19': [91.6000,91.79000,'usado'],
        'canal 20': [91.8000,91.99000,'libre'],
    
        }

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
                    
                    print(data_canal.loc[idmax])
                    print(maxim)
            else:
                print('No hay interferencia en el ' + str(key))
            

def segunda_iteracion():

    f_min = 92e6
    f_max = 96e6
    veces=50
    
    canales ={
        'canal 21': [92.0000,92.19000,'usado'],
        'canal 22': [92.2000,92.39000,'libre'],
        'canal 23': [92.4000,92.59000,'usado'],
        'canal 24': [92.6000,92.79000,'libre'],
        'canal 25': [92.8000,92.99000,'libre'],
        'canal 26': [93.0000,93.19000,'libre'],
        'canal 27': [93.2000,93.39000,'usado'],
        'canal 28': [93.4000,93.59000,'libre'],
        'canal 29': [93.6000,93.79000,'usado'],
        'canal 30': [93.8000,93.99000,'libre'],
        'canal 31': [94.0000,94.19000,'usado'],
        'canal 32': [94.2000,94.39000,'libre'],
        'canal 33': [94.4000,94.59000,'libre'],
        'canal 34': [94.6000,94.79000,'libre'],
        'canal 35': [94.8000,94.99000,'usado'],
        'canal 36': [95.0000,95.19000,'libre'],
        'canal 37': [95.2000,95.39000,'usado'],
        'canal 38': [95.4000,95.59000,'libre'],
        'canal 39': [95.6000,95.79000,'usado'],
        'canal 40': [95.8000,95.99000,'libre'],
        } 

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
            if corr < 0.5 and rmse >10:
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                if maxim > -25 and maxim < 200:
                
                    print(data_canal.loc[idmax])
                    print(maxim)
            else:
                print('No hay interferencia en el ' + str(key))
            
def tercera_iteracion():

    f_min = 96e6
    f_max = 100e6
    veces=50
    
    canales ={
        'canal 41': [96.0000,96.19000,'usado'],
        'canal 42': [96.2000,96.39000,'libre'],
        'canal 43': [96.4000,96.59000,'usado'],
        'canal 44': [96.6000,96.79000,'libre'],
        'canal 45': [96.8000,96.99000,'usado'],
        'canal 46': [97.0000,97.19000,'libre'],
        'canal 47': [97.2000,97.39000,'usado'],
        'canal 48': [97.4000,97.59000,'libre'],
        'canal 49': [97.6000,97.79000,'usado'],
        'canal 50': [97.8000,97.99000,'libre'],
        'canal 51': [98.0000,98.19000,'usado'],
        'canal 52': [98.2000,98.39000,'libre'],
        'canal 53': [98.4000,98.59000,'libre'],
        'canal 54': [98.6000,98.79000,'libre'],
        'canal 55': [98.8000,98.99000,'usado'],
        'canal 56': [99.0000,99.19000,'libre'],
        'canal 57': [99.2000,99.39000,'usado'],
        'canal 58': [99.4000,99.59000,'libre'],
        'canal 59': [99.6000,99.79000,'usado'],
        'canal 60': [99.8000,99.99000,'libre'],
        }  

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
            if corr < 0.5 and rmse >10:
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                if maxim > -20 and maxim < 200:
                    print(data_canal.loc[idmax])
                    print(maxim)
            else:
                print('No hay interferencia en el ' + str(key))



def cuarta_iteracion():

    f_min = 100e6
    f_max = 104e6
    veces=50
    
    canales ={
        'canal 61': [100.0000,100.19000,'usado'],
        'canal 62': [100.2000,100.39000,'libre'],
        'canal 63': [100.4000,100.59000,'usado'],
        'canal 64': [100.6000,100.79000,'libre'],
        'canal 65': [100.8000,100.99000,'libre'],
        'canal 66': [101.0000,101.19000,'libre'],
        'canal 67': [101.2000,101.39000,'usado'],
        'canal 68': [101.4000,101.59000,'libre'],
        'canal 69': [101.6000,101.79000,'usado'],
        'canal 70': [101.8000,101.99000,'libre'],
        'canal 71': [102.0000,102.19000,'usado'],
        'canal 72': [102.2000,102.39000,'libre'],
        'canal 73': [102.4000,102.59000,'usado'],
        'canal 74': [102.6000,102.79000,'libre'],
        'canal 75': [102.8000,102.99000,'usado'],
        'canal 76': [103.0000,103.19000,'libre'],
        'canal 77': [103.2000,103.39000,'usado'],
        'canal 78': [103.4000,103.59000,'libre'],
        'canal 79': [103.6000,103.79000,'usado'],
        'canal 80': [103.8000,103.99000,'libre'],
        }  

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
            if corr < 0.5 and rmse >10:
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                if maxim > -20 and maxim < 200:
                    print(data_canal.loc[idmax])
                    print(maxim)
            else:
                print('No hay interferencia en el ' + str(key))

def quinta_iteracion():

    f_min = 104e6
    f_max = 108e6
    veces=50
    
    canales ={
        'canal 81': [104.0000,104.19000,'usado'],
        'canal 82': [104.2000,104.39000,'libre'],
        'canal 83': [104.4000,104.59000,'usado'],
        'canal 84': [104.6000,104.79000,'libre'],
        'canal 85': [104.8000,104.99000,'usado'],
        'canal 86': [105.0000,105.19000,'libre'],
        'canal 87': [105.2000,105.39000,'usado'],
        'canal 88': [105.4000,105.59000,'libre'],
        'canal 89': [105.6000,105.79000,'usado'],
        'canal 90': [105.8000,105.99000,'libre'],
        'canal 91': [106.0000,106.19000,'usado'],
        'canal 92': [106.2000,106.39000,'libre'],
        'canal 93': [106.4000,106.59000,'usado'],
        'canal 94': [106.6000,106.79000,'libre'],
        'canal 95': [106.8000,106.99000,'usado'],
        'canal 96': [107.0000,107.19000,'libre'],
        'canal 97': [107.2000,107.39000,'usado'],
        'canal 98': [107.4000,107.59000,'libre'],
        'canal 99': [107.6000,107.79000,'usado'],
        'canal 100': [107.8000,107.99000,'libre'],
        }  

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
            if corr < 0.5 and rmse >10:
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                if maxim > -20 and maxim < 200:
                    print(data_canal.loc[idmax])
                    print(maxim)
            else:
                print('No hay interferencia en el ' + str(key))




if __name__ == "__main__":
    primera_iteracion()
    segunda_iteracion()
    tercera_iteracion()
    cuarta_iteracion()
    quinta_iteracion()




