#define the CMUnet loss calculation
#__author__ = 'Haiyang Liu'

def get_loss(output,target,mask,config):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {'final': output}
    pass
    return loss