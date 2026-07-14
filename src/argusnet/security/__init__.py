"""Device authentication, signed envelopes, and transport security.

This package is the trust boundary for data entering ArgusNet over untrusted
transports (MQTT, non-loopback gRPC). It never touches fusion math.
"""
