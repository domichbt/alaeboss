import numpy as np
from numpy.typing import ArrayLike
from alaeboss import LinearRegressor

from LSS.imaging.densvar import read_systematic_maps

def get_imweight_alt_v1p5(dd: ArrayLike, rd: ArrayLike, zmin: float, zmax: float, reg: str, fit_maps: list[str], use_maps: list[str], plotr: bool = False, zcol: str = 'Z', sys_tab=None, wtmd: str = 'fracz', figname: str = 'temp.png'):

    assert (set(fit_maps) == set(use_maps)), "Parameter `fit_maps` is included for retrocompatibility but cannot be different from `use_maps` in this implementation."

    data_mask = (dd[zcol] > zmin) & (dd[zcol] < zmax) & (dd['PHOTSYS'] == reg)
    random_mask = (rd['PHOTSYS'] == reg) # always select randoms on region at least
    if wtmd == "clus":
        random_mask &= ((rd[zcol] > zmin) & (rd[zcol] < zmax))
    data_selected = dd[data_mask]
    randoms_selected = rd[random_mask]
    data_syst, rand_syst = read_systematic_maps(data_selected['RA'], data_selected['DEC'], randoms_selected['RA'], randoms_selected['DEC'], sys_tab=sys_tab)

    column_names = list(dd.dtype.names)
    random_weights = np.ones(len(randoms_selected))

    if wtmd == 'fracz':
        print('using 1/FRACZ_TILELOCID based completeness weights')
        data_weights = 1/data_selected['FRACZ_TILELOCID']
        if 'FRAC_TLOBS_TILES' in column_names:
            print('using FRAC_TLOBS_TILES')
            data_weights *= 1/data_selected['FRAC_TLOBS_TILES']

    elif wtmd == 'wt':
        data_weights = data_selected['WEIGHT']
        random_weights = rd['WEIGHT']

    elif wtmd == 'wtfkp':
        data_weights = data_selected['WEIGHT']*data_selected['WEIGHT_FKP']
        random_weights = rd['WEIGHT']*rd['WEIGHT_FKP']

    elif wtmd == 'wt_comp':
        data_weights = data_selected['WEIGHT_COMP']
    
    elif wtmd == 'clus':
        data_weights = data_selected['WEIGHT']*data_selected['WEIGHT_FKP']/data_selected['WEIGHT_SYS']
        random_weights = randoms_selected['WEIGHT']*randoms_selected['WEIGHT_FKP']/randoms_selected['WEIGHT_SYS']

    if ('WEIGHT_ZFAIL' in column_names) and (wtmd != 'clus'):
        data_weights *= data_selected['WEIGHT_ZFAIL']
    
    templates = {name:(data_syst[name], rand_syst[name]) for name in use_maps}

    regressor = LinearRegressor(data_weights=data_weights, random_weights=random_weights, templates=templates, loglevel='INFO')
    regressor.cut_outliers(tail=0.5)
    regressor.prepare(nbins=10)
    result = regressor.regress_minuit()
    
    if result is None:
        raise RuntimeError("The minimization failed unexpectedly. No weights available. Please investigate failure status from the traceback.")
    regressor.logger.info("Minimization results are %s", result)

    # plotting
    if plotr:
        raise NotImplementedError("Results plotting is not implemented")

    weights = np.ones(shape=(len(dd)), dtype=float)
    weights[np.flatnonzero(data_mask)[regressor.good_values_data]] = regressor.export_weights()
    return weights


def get_imweight_alt(dd,rd,zmin,zmax,reg,fit_maps,use_maps,plotr=False,zcol='Z',sys_tab=None,wtmd='fracz',figname='temp.png',modoutname='temp.txt',logger=None):
    """
    Drop-in replacement for get_imweight (tag DR2-v2) using autodiff fit. This does NOT support the plotting or having fit_maps different from use_maps and will raise an error if these options are used.
    """
    assert (set(fit_maps) == set(use_maps)), "Parameter `fit_maps` is included for retrocompatibility but cannot be different from `use_maps` in this implementation."
    import sys
    sys.path.append("/global/homes/d/dchebat/imsys/alaeboss/")
    from alaeboss import LinearRegressor
    import jax
    jax.config.update('jax_enable_x64', True)

    import LSS.common_tools as common

    sel = dd[zcol] > zmin
    sel &= dd[zcol] < zmax
    if reg == 'N' or reg == 'S':
        sel &= dd['PHOTSYS'] == reg
        selr = rd['PHOTSYS'] == reg
    elif 'DES' in reg:
        inDES = common.select_regressis_DES(dd)
        inDESr = common.select_regressis_DES(rd)
        if reg == 'DES':
            sel &= inDES
            selr = inDESr
        if reg == 'SnotDES':
            sel &= dd['PHOTSYS'] == 'S'
            sel &= ~inDES
            selr = rd['PHOTSYS'] == 'S'
            selr &= ~inDESr

    else:
        print('other regions not currently supported')
        return 'Exiting due to critical error with region'
 
    dds = dd[sel]

    if wtmd == 'clus':
        selr &= rd[zcol] > zmin
        selr &= rd[zcol] < zmax

    rd = rd[selr]

    #-- Dictionaries containing all different systematic values
    data_syst, rand_syst = read_systematic_maps(dds['RA'],dds['DEC'],rd['RA'],rd['DEC'],sys_tab=sys_tab)
    #print(data_syst.keys)
    cols = list(dd.dtype.names)
    weights_ran = np.ones(len(rd))
    if wtmd == 'fracz':
        common.printlog('using 1/FRACZ_TILELOCID based completeness weights',logger)
        wts = 1/dds['FRACZ_TILELOCID']
        if 'FRAC_TLOBS_TILES' in cols:
            common.printlog('using FRAC_TLOBS_TILES',logger)
            wts *= 1/dds['FRAC_TLOBS_TILES']
    if wtmd == 'wt':
        wts = dds['WEIGHT']
        weights_ran = rd['WEIGHT']
    if wtmd == 'wtfkp':
        wts = dds['WEIGHT']*dds['WEIGHT_FKP']
        weights_ran = rd['WEIGHT']*rd['WEIGHT_FKP']

    if wtmd == 'clus':
        wts = dds['WEIGHT']*dds['WEIGHT_FKP']/dds['WEIGHT_SYS']
        weights_ran = rd['WEIGHT']*rd['WEIGHT_FKP']/rd['WEIGHT_SYS']
    if wtmd == 'wt_comp':
        wts = dds['WEIGHT_COMP']

    if 'WEIGHT_ZFAIL' in cols and wtmd != 'clus':
        wts *= dds['WEIGHT_ZFAIL']

    data_we = wts
    rand_we = weights_ran
    #-- Create fitter object and add the systematic maps we want 
    templates = {name:(data_syst[name], rand_syst[name]) for name in use_maps}
    regressor = LinearRegressor(data_weights=data_we, random_weights=rand_we, templates=templates)
    regressor.cut_outliers(tail=0.5) 

    #-- Perform global fit
    nbins=10
    regressor.prepare(nbins=nbins)
    #for name in fit_maps:
    #    print(name,len(s.data_syst[name]),len(s.data_we))
    pars_dict = regressor.regress_minuit()
    common.printlog(str(regressor.coefficients),logger)
    common.printlog(str(list(regressor.coefficients)),logger)
    # pars_dict = {}
    common.printlog('writing to '+modoutname,logger)
    fo = open(modoutname,'w')
    for par_name, par_value in pars_dict.items():
        fo.write(str(par_name)+' '+str(par_value)+'\n')
    fo.close()
    if plotr:
        raise NotImplementedError("Results plotting is not implemented")
        # #s.plot_overdensity(pars=[None, s.best_pars], ylim=[0.7, 1.3])#, title=f'{sample_name}: global fit')
        # common.printlog('saving figure to '+figname,logger)
        # regressor.plot_overdensity(pars=[None, pars_dict], ylim=[0.7, 1.3])
        # plt.savefig(figname)
        # plt.clf()
        # #plt.show()
    #-- Get weights for global fit
    #data_weightsys_global = 1/s.get_model(s.best_pars, data_syst)
    data_weightsys_global = regressor.export_weights()
    wsysl = np.ones(len(dd))
    wsysl[sel] = data_weightsys_global 
    del data_syst
    del rand_syst
    return wsysl

