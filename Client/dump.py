
def dump_si(si):
    #global df
    out_fields["Frame"] = frame_num
    out_fields["Timestamp"] = data.timestamp
    out_fields["UTC Time"] = time.time_ns()

    dump_path = file_path + "/" + "dumps/"
    os.mkdir(dump_path)

    file_to_dump = open(dump_path+ "/" + str(out_fields["Timestamp"]), 'wb')
    pickle.dump(si, file_to_dump)

    file_to_dump.close()
    print("dumped: " + str(out_fields["Timestamp"]))
