import synapseclient
import synapseutils


if __name__ == "__main__":
    target_path = '/path/to/brats23'
    syn = synapseclient.Synapse() 
    syn.login(authToken='') # your token
    files = synapseutils.syncFromSynapse(syn, 'syn51156910', path=target_path)