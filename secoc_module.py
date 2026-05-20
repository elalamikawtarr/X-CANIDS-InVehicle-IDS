# MODULE DE SIMULATION SecOC 
# Ce script implémente une signature cryptographique HMAC-SHA256 tronquée.
# Il garantit l'Authenticité (via la clé secrète) et la Fraîcheur (via le FV)
# pour empêcher les attaques par usurpation (Masquerade) et par rejeu (Replay).

import hmac
import hashlib

def simuler_secoc(can_id, data, secret_key, freshness_value):
    # 1. Préparation du message (ID + Data + Compteur de fraîcheur)
    message = f"{can_id}{data}{freshness_value}".encode()
    
    # 2. Génération du MAC 
    mac = hmac.new(secret_key.encode(), message, hashlib.sha256).hexdigest()
    mac_tronque = mac[:8] 
    
    return {
        "CAN_ID": can_id,
        "Payload": data,
        "FV": freshness_value,
        "MAC": mac_tronque
    }


ma_cle_secrete = "CLE_PFE_2024"
msg_protege = simuler_secoc("0x112", "75km/h", ma_cle_secrete, 101)

print("     PROTECTION ---")
print(f"Message sécurisé envoyé : {msg_protege}")
