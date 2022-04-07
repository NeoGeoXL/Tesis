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

def setup1(f_min, f_max,veces):
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
    
    psd_total=np.empty([len*14,veces])

    return rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total

def setup2(f_min, f_max,veces):
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
    
    psd_total=np.empty([len*17,veces])

    return rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total


def setup3(f_min, f_max,veces):
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
    
    psd_total=np.empty([len*30,veces])

    return rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total


def setup4(f_min, f_max,veces):
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
    
    psd_total=np.empty([len*25,veces])

    return rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total


def setup5(f_min, f_max,veces):
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
    
    psd_total=np.empty([len*25,veces])

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
    #psd_total=np.reshape(len(psd_series),1)
    #print(psd_total.shape)
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

def detection_limit(n,umbral=-45,constante=-45):
    #umbral = -45
    if n <= umbral: 
        return constante
    else:
        return n

def ploteo_senal_comparacion_y_referencia(data_canal,senal_referencia,senal_comparacion,key):
    #print(senal_comparacion) #82,1
    #print(senal_referencia) #82,
    #senal_comparacion = senal_comparacion.squeeze()
    #print(' el shape de senal comparacion: ')
    #print(senal_comparacion.shape)
    #print(senal_comparacion.shape) #82,1
    #print(senal_referencia.shape) #82,




    data_canal_freqs=data_canal['Frecuencia']
    data_canal_freqs = data_canal_freqs.to_numpy()
    referencia_numpy = senal_referencia.to_numpy()
    comparacion_numpy = senal_comparacion.to_numpy()

    referencia_signal = np.stack((data_canal_freqs,  referencia_numpy), axis=1)
    comparacion_signal = np.stack((data_canal_freqs,  comparacion_numpy), axis=1)
    #print(referencia_signal)
    #print(comparacion_signal)

    plt.subplot(2,1,1)
    plt.plot(referencia_signal[:,0],referencia_signal[:,1])
    plt.title('Senal de referencia')
    #plt.xlabel('Frecuencia [MHz]')
    plt.ylabel('Potencia [dB]')
    plt.subplot(2,1,2)
    plt.plot(comparacion_signal[:,0],comparacion_signal[:,1])
    plt.title('Senal de comparacion')
    plt.xlabel('Frecuencia [MHz]')
    plt.ylabel('Potencia [dB]')

    plt.suptitle('Senales de referencia y comparacion del: {}'.format(key))
    plt.show()

def correlacion(senal_referencia,senal_comparacion):
    pass



def comparacion(data_canal,senal_referencia,senal_comparacion,key):
    senal_comparacion=senal_comparacion.squeeze()
    #ploteo_senal_comparacion_y_referencia(data_canal,senal_referencia,senal_comparacion,key)

    '''    senal_referencia_numpy = senal_referencia.to_numpy()
    senal_comparacion_numpy = senal_comparacion.to_numpy()
    correlacion_signal = np.stack((senal_referencia_numpy,  senal_comparacion_numpy), axis=1)
    correlacion_df = pd.DataFrame(correlacion_signal,columns=['Referencia','Comparacion'])

    print(correlacion_df)'''

    '''senal_referencia_numpy = senal_referencia.to_numpy()
    senal_comparacion_numpy = senal_comparacion.to_numpy()

    corre = np.corrcoef(senal_referencia_numpy,senal_comparacion_numpy)
    print('La correlacion con numpy es : {}'.format(corre))'''
    
    corr=senal_referencia.corr(senal_comparacion)
    #print(corr)
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
            if data_canal['Potencia'].max() < -35:
                umbral = data_canal['Potencia'].max()
                senal_referencia=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))
            else: 
                umbral = data_canal['Potencia'].min()
                senal_referencia=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))

                
    #print('El umbral es: '+ str(umbral)+' dBm'+' del '+str(key))
    return umbral, senal_referencia  

def minima_senal_detectable_canal(data):
    senal_referencia = data['Potencia'].apply(detection_limit,args=(-29,-29))   #-29 para primera iteracion
    #plt.plot(senal_referencia)
    return senal_referencia

#Senal referencia = Senal leida con el sdr y con la tranmision no deseada
#Senal comparacion = Senal creada a partir de la longitud del df y con los valores por defecto del umbral

def crear_senal_comparacion(senal_referencia,umbral):
    senal_comparacion=np.empty(senal_referencia.shape[0])
    senal_comparacion[:] = umbral
    '''senal_comparacion = senal_comparacion.squeeze()
    print(' el shape de senal comparacion: ')
    print(senal_comparacion.shape)'''
    senal_comparacion=pd.DataFrame(senal_comparacion)
    #print(senal_comparacion)
    return senal_comparacion
    
def signal_coherence(senal_referencia,senal_comparacion):
    f, Cxy = signal.coherence(senal_referencia,senal_comparacion)
    plt.plot(f, Cxy)
    return Cxy


def filtrado_canal(data,f_min_canal,f_max_canal):
    data_canal=canal_filter(data,f_min_canal,f_max_canal)
    #umbral=data_canal['Potencia'].max()
    #senal_referencia=data_canal['Potencia'].apply(detection_limit,args=(umbral,umbral))
    return data_canal


def comparacion_senales(data_canal,senal_referencia,senal_comparacion,key):

    if senal_comparacion.shape[0] != senal_referencia.shape[0]:
        senal_comparacion=senal_comparacion[0:senal_referencia.shape[0]]
    corr,rmse = comparacion(data_canal,senal_referencia,senal_comparacion,key)     #compararmos la senal con la misma solo para probar 
    #coherencia = signal_coherence(senal_referencia,senal_comparacion)
    #print('La coherencia es: '+ str(coherencia))
    return corr,rmse


    
    
def procesamiento1(f_min,f_max,canales):
    veces=1
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup1(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    #umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            data_canal=filtrado_canal(data,f_min_canal,f_max_canal)
            senal_referencia=minima_senal_detectable_canal(data_canal)
            senal_comparacion = crear_senal_comparacion(senal_referencia,-29) 
            #Faltaria la senal de referencia que tiene todos los valores iguales a -45
            
            corr, rmse = comparacion_senales(data_canal,senal_referencia,senal_comparacion,key)
            #data_canal.to_csv(r'C:\Users\ggarc\Desktop\Tesis\matrizfm')

            '''maxim=data_canal['Potencia'].max()
            idmax=data_canal['Potencia'].idxmax()
            parasita = data_canal.loc[idmax]
            max_freq=parasita['Frecuencia']
            max_pot=parasita['Potencia']
            print(parasita)
            print(max_freq)
            print(max_pot)
            espuria={
                    'Frecuencia':max_freq,
                    'Potencia':max_pot,
                    }
            print(espuria)
            #par=parasita.to_dict()
            #print(par)'''


            if corr < 0.25 and rmse > 0.001 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -200 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    max_freq=parasita['Frecuencia']
                    max_pot=parasita['Potencia']
                    #print(parasita)
                    #print(max_freq)
                    #print(max_pot)
                    plt.subplot(3,1,1)
                    plt.plot(data_canal['Frecuencia'],data_canal['Potencia'])
                    plt.title('Senal Electromagnetica de '+str(key))
                    plt.subplot(3,1,2)
                    plt.plot(senal_referencia)
                    plt.title('Senal Referencia')
                    plt.subplot(3,1,3)
                    plt.plot(senal_comparacion)
                    plt.title('Senal Comparacion')
                    plt.show()

                    espuria={
                        'Frecuencia':max_freq,
                         'Potencia':max_pot,
                    }
                    print('La transmision no deseada se encuentra en {} con una potencia de: {}'.format(max_freq,max_pot))
                    return data_canal, espuria , 1 #El 1 indica que se encontro una espuria
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                print('El rmse es: '+ str(rmse))
                print('La correlacion es ' + str(corr))
                print('*'*50)
                espuria={
                        'Frecuencia':0,
                         'Potencia':0,
                    }
   

    '''CURRENT_DIR = pathlib.Path().resolve()  
    #print(CURRENT_DIR)
    IMAGE_DIR=CURRENT_DIR.joinpath("app","static","images","fm")
    #print(IMAGE_DIR)
    #print(IMAGE_DIR.exists())
    IMAGE_DIR = str(IMAGE_DIR)'''


    data_numpy=data.to_numpy()                  
    plt.plot(data_numpy[:,0],data_numpy[:,1])
    plt.title('Grafica de la iteracion primera del espectro de Radio TV VHF de: {} a {} [MHz]'.format(f_min/1e6,f_max/1e6))
    plt.xlabel('Frecuencia [MHZ]')
    plt.ylabel('Potencia [dB]')
    #plt.savefig('\espectro_iteracion_{}.png')
    #plt.show()
    



    return data, espuria, 0 # El 0 indica que no se encontro una espuria

def procesamiento2(f_min,f_max,canales):
    veces=1
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup2(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    #umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            data_canal=filtrado_canal(data,f_min_canal,f_max_canal)
            senal_referencia=minima_senal_detectable_canal(data_canal)
            senal_comparacion = crear_senal_comparacion(senal_referencia,-29) 
            #Faltaria la senal de referencia que tiene todos los valores iguales a -45
            
            corr, rmse = comparacion_senales(data_canal,senal_referencia,senal_comparacion,key)
            #data_canal.to_csv(r'C:\Users\ggarc\Desktop\Tesis\matrizfm')

            '''maxim=data_canal['Potencia'].max()
            idmax=data_canal['Potencia'].idxmax()
            parasita = data_canal.loc[idmax]
            max_freq=parasita['Frecuencia']
            max_pot=parasita['Potencia']
            print(parasita)
            print(max_freq)
            print(max_pot)
            espuria={
                    'Frecuencia':max_freq,
                    'Potencia':max_pot,
                    }
            print(espuria)
            #par=parasita.to_dict()
            #print(par)'''


            if corr < 0.25 and rmse > 0.001 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -200 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    max_freq=parasita['Frecuencia']
                    max_pot=parasita['Potencia']
                    #print(parasita)
                    #print(max_freq)
                    #print(max_pot)
                    plt.subplot(3,1,1)
                    plt.plot(data_canal['Frecuencia'],data_canal['Potencia'])
                    plt.title('Senal Electromagnetica de '+str(key))
                    plt.subplot(3,1,2)
                    plt.plot(senal_referencia)
                    plt.title('Senal Referencia')
                    plt.subplot(3,1,3)
                    plt.plot(senal_comparacion)
                    plt.title('Senal Comparacion')
                    plt.show()

                    espuria={
                        'Frecuencia':max_freq,
                         'Potencia':max_pot,
                    }
                    print('La transmision no deseada se encuentra en {} con una potencia de: {}'.format(max_freq,max_pot))
                    return data_canal, espuria , 1 #El 1 indica que se encontro una espuria
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                print('El rmse es: '+ str(rmse))
                print('La correlacion es ' + str(corr))
                print('*'*50)
                espuria={
                        'Frecuencia':0,
                         'Potencia':0,
                    }
   

    '''CURRENT_DIR = pathlib.Path().resolve()  
    #print(CURRENT_DIR)
    IMAGE_DIR=CURRENT_DIR.joinpath("app","static","images","fm")
    #print(IMAGE_DIR)
    #print(IMAGE_DIR.exists())
    IMAGE_DIR = str(IMAGE_DIR)'''


    data_numpy=data.to_numpy()                  
    plt.plot(data_numpy[:,0],data_numpy[:,1])
    plt.title('Grafica de la iteracion segunda del espectro de Radio TV VHF de: {} a {} [MHz]'.format(f_min/1e6,f_max/1e6))
    plt.xlabel('Frecuencia [MHZ]')
    plt.ylabel('Potencia [dB]')
    #plt.savefig('\espectro_iteracion_{}.png')
    #plt.show()
    



    return data, espuria, 0 # El 0 indica que no se encontro una espuria

def procesamiento3(f_min,f_max,canales):
    veces=1
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup3(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    #umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            data_canal=filtrado_canal(data,f_min_canal,f_max_canal)
            senal_referencia=minima_senal_detectable_canal(data_canal)
            senal_comparacion = crear_senal_comparacion(senal_referencia,-29) 
            #Faltaria la senal de referencia que tiene todos los valores iguales a -45
            
            corr, rmse = comparacion_senales(data_canal,senal_referencia,senal_comparacion,key)
            #data_canal.to_csv(r'C:\Users\ggarc\Desktop\Tesis\matrizfm')

            '''maxim=data_canal['Potencia'].max()
            idmax=data_canal['Potencia'].idxmax()
            parasita = data_canal.loc[idmax]
            max_freq=parasita['Frecuencia']
            max_pot=parasita['Potencia']
            print(parasita)
            print(max_freq)
            print(max_pot)
            espuria={
                    'Frecuencia':max_freq,
                    'Potencia':max_pot,
                    }
            print(espuria)
            #par=parasita.to_dict()
            #print(par)'''


            if corr < 0.25 and rmse > 0.01 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -200 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    max_freq=parasita['Frecuencia']
                    max_pot=parasita['Potencia']
                    #print(parasita)
                    #print(max_freq)
                    #print(max_pot)
                    plt.subplot(3,1,1)
                    plt.plot(data_canal['Frecuencia'],data_canal['Potencia'])
                    plt.title('Senal Electromagnetica de '+str(key))
                    plt.subplot(3,1,2)
                    plt.plot(senal_referencia)
                    plt.title('Senal Referencia')
                    plt.subplot(3,1,3)
                    plt.plot(senal_comparacion)
                    plt.title('Senal Comparacion')
                    plt.show()

                    espuria={
                        'Frecuencia':max_freq,
                         'Potencia':max_pot,
                    }
                    print('La transmision no deseada se encuentra en {} con una potencia de: {}'.format(max_freq,max_pot))
                    return data_canal, espuria , 1 #El 1 indica que se encontro una espuria
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                print('El rmse es: '+ str(rmse))
                print('La correlacion es ' + str(corr))
                print('*'*50)
                espuria={
                        'Frecuencia':0,
                         'Potencia':0,
                    }
   

    '''CURRENT_DIR = pathlib.Path().resolve()  
    #print(CURRENT_DIR)
    IMAGE_DIR=CURRENT_DIR.joinpath("app","static","images","fm")
    #print(IMAGE_DIR)
    #print(IMAGE_DIR.exists())
    IMAGE_DIR = str(IMAGE_DIR)'''


    data_numpy=data.to_numpy()                  
    plt.plot(data_numpy[:,0],data_numpy[:,1])
    plt.title('Grafica de la iteracion primera del espectro de Radio TV VHF de: {} a {} [MHz]'.format(f_min/1e6,f_max/1e6))
    plt.xlabel('Frecuencia [MHZ]')
    plt.ylabel('Potencia [dB]')
    #plt.savefig('\espectro_iteracion_{}.png')
    #plt.show()
    



    return data, espuria, 0 # El 0 indica que no se encontro una espuria

def procesamiento4(f_min,f_max,canales):
    veces=1
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup4(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    #umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            data_canal=filtrado_canal(data,f_min_canal,f_max_canal)
            senal_referencia=minima_senal_detectable_canal(data_canal)
            senal_comparacion = crear_senal_comparacion(senal_referencia,-29) 
            #Faltaria la senal de referencia que tiene todos los valores iguales a -45
            
            corr, rmse = comparacion_senales(data_canal,senal_referencia,senal_comparacion,key)
            #data_canal.to_csv(r'C:\Users\ggarc\Desktop\Tesis\matrizfm')

            '''maxim=data_canal['Potencia'].max()
            idmax=data_canal['Potencia'].idxmax()
            parasita = data_canal.loc[idmax]
            max_freq=parasita['Frecuencia']
            max_pot=parasita['Potencia']
            print(parasita)
            print(max_freq)
            print(max_pot)
            espuria={
                    'Frecuencia':max_freq,
                    'Potencia':max_pot,
                    }
            print(espuria)
            #par=parasita.to_dict()
            #print(par)'''


            if corr < 0.25 and rmse > 0.001 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -200 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    max_freq=parasita['Frecuencia']
                    max_pot=parasita['Potencia']
                    #print(parasita)
                    #print(max_freq)
                    #print(max_pot)
                    plt.subplot(3,1,1)
                    plt.plot(data_canal['Frecuencia'],data_canal['Potencia'])
                    plt.title('Senal Electromagnetica de '+str(key))
                    plt.subplot(3,1,2)
                    plt.plot(senal_referencia)
                    plt.title('Senal Referencia')
                    plt.subplot(3,1,3)
                    plt.plot(senal_comparacion)
                    plt.title('Senal Comparacion')
                    plt.show()

                    espuria={
                        'Frecuencia':max_freq,
                         'Potencia':max_pot,
                    }
                    print('La transmision no deseada se encuentra en {} con una potencia de: {}'.format(max_freq,max_pot))
                    return data_canal, espuria , 1 #El 1 indica que se encontro una espuria
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                print('El rmse es: '+ str(rmse))
                print('La correlacion es ' + str(corr))
                print('*'*50)
                espuria={
                        'Frecuencia':0,
                         'Potencia':0,
                    }
   

    '''CURRENT_DIR = pathlib.Path().resolve()  
    #print(CURRENT_DIR)
    IMAGE_DIR=CURRENT_DIR.joinpath("app","static","images","fm")
    #print(IMAGE_DIR)
    #print(IMAGE_DIR.exists())
    IMAGE_DIR = str(IMAGE_DIR)'''


    data_numpy=data.to_numpy()                  
    plt.plot(data_numpy[:,0],data_numpy[:,1])
    plt.title('Grafica de la iteracion primera del espectro de Radio TV VHF de: {} a {} [MHz]'.format(f_min/1e6,f_max/1e6))
    plt.xlabel('Frecuencia [MHZ]')
    plt.ylabel('Potencia [dB]')
    #plt.savefig('\espectro_iteracion_{}.png')
    #plt.show()
    



    return data, espuria, 0 # El 0 indica que no se encontro una espuria

def procesamiento5(f_min,f_max,canales):
    veces=1
    rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total= setup5(f_min, f_max,veces)
    data=readsdr(rate_best, freqs, nfreq, npsd_res, npsd_avg, nsamp, nfreq_spec, samples, psd_array, freq_array, relative_power_array, psd_total,veces)
    #umbral,senal_referencia=minimun_signal_detectable(canales,data)

    for key in canales:
        values=canales[key]
        condicicon=values[2]
        if condicicon=='libre':
            
            f_min_canal=values[0]
            f_max_canal=values[1]

            data_canal=filtrado_canal(data,f_min_canal,f_max_canal)
            senal_referencia=minima_senal_detectable_canal(data_canal)
            senal_comparacion = crear_senal_comparacion(senal_referencia,-29) 
            #Faltaria la senal de referencia que tiene todos los valores iguales a -45
            
            corr, rmse = comparacion_senales(data_canal,senal_referencia,senal_comparacion,key)
            #data_canal.to_csv(r'C:\Users\ggarc\Desktop\Tesis\matrizfm')

            '''maxim=data_canal['Potencia'].max()
            idmax=data_canal['Potencia'].idxmax()
            parasita = data_canal.loc[idmax]
            max_freq=parasita['Frecuencia']
            max_pot=parasita['Potencia']
            print(parasita)
            print(max_freq)
            print(max_pot)
            espuria={
                    'Frecuencia':max_freq,
                    'Potencia':max_pot,
                    }
            print(espuria)
            #par=parasita.to_dict()
            #print(par)'''


            if corr < 0.25 and rmse > 0.001 :
                maxim=data_canal['Potencia'].max()
                idmax=data_canal['Potencia'].idxmax()
                #print(rmse)
                if maxim > -200 and maxim < 200:
                    parasita = data_canal.loc[idmax]
                    max_freq=parasita['Frecuencia']
                    max_pot=parasita['Potencia']
                    #print(parasita)
                    #print(max_freq)
                    #print(max_pot)
                    plt.subplot(3,1,1)
                    plt.plot(data_canal['Frecuencia'],data_canal['Potencia'])
                    plt.title('Senal Electromagnetica de '+str(key))
                    plt.subplot(3,1,2)
                    plt.plot(senal_referencia)
                    plt.title('Senal Referencia')
                    plt.subplot(3,1,3)
                    plt.plot(senal_comparacion)
                    plt.title('Senal Comparacion')
                    plt.show()

                    espuria={
                        'Frecuencia':max_freq,
                         'Potencia':max_pot,
                    }
                    print('La transmision no deseada se encuentra en {} con una potencia de: {}'.format(max_freq,max_pot))
                    return data_canal, espuria , 1 #El 1 indica que se encontro una espuria
                    #Para la app web mandas un diccionario con 1 si hay una frecuencia parasita y el valor de la frecuencia y 0 si no hay frecuencia parasita
            else:
                print('No hay interferencia en el ' + str(key))
                print('El rmse es: '+ str(rmse))
                print('La correlacion es ' + str(corr))
                print('*'*50)
                espuria={
                        'Frecuencia':0,
                         'Potencia':0,
                    }
   

    '''CURRENT_DIR = pathlib.Path().resolve()  
    #print(CURRENT_DIR)
    IMAGE_DIR=CURRENT_DIR.joinpath("app","static","images","fm")
    #print(IMAGE_DIR)
    #print(IMAGE_DIR.exists())
    IMAGE_DIR = str(IMAGE_DIR)'''


    data_numpy=data.to_numpy()                  
    plt.plot(data_numpy[:,0],data_numpy[:,1])
    plt.title('Grafica de la iteracion primera del espectro de Radio TV VHF de: {} a {} [MHz]'.format(f_min/1e6,f_max/1e6))
    plt.xlabel('Frecuencia [MHZ]')
    plt.ylabel('Potencia [dB]')
    #plt.savefig('\espectro_iteracion_{}.png')
    #plt.show()
    



    return data, espuria, 0 # El 0 indica que no se encontro una espuria

def procesamiento_diccionarios(datos):
    datos.set_index('Frecuencia',inplace=True)
    datos=datos.rename_axis('Frecuencia')
    datos_potencia = datos['Potencia'].tolist()
    datos=datos.to_dict(orient='split')
    del datos['columns']
    espectro = {
        'Frecuencia': datos['index'],
        'Potencia': datos_potencia,
    }
    return espectro
    
