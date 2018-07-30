from settings import setup_logging
logger = setup_logging()

import requests
import json

class session(object):
    def __init__(self, url):
        self.url = url

    def get(self, params=None, **kwargs):
        r = requests.get(self.url, params, **kwargs)
        if r.status_code != 200:
            logger.error("Requests failed, HTTP code: {0}".format(r.status_code))
        return r

class mast_api(object):

    def __init__(self):
        self.missions = ["kepler", "k2"]
        self.tables_kep = ["data_search", "kepler_fov", "kic10", "kgmatch", \
                           "confirmed_planets", "published_planets", "koi", "ffi"]
        self.tables_k2 = ["epic", "data_search", "published_planets", "ffi"]
        self.other_missions = ["hst", "hsc_sum", "hsc", "iue", "hut", "euve", "fuse", \
                               "uit", "wuppe", "befs", "tues", "imaps", "hlsp", "pointings", \
                               "copernicus", "hpol", "vlafirst", "xmm-om", "swift_uvot"]
        self.outputs = ["HTML_Table", "Excel_Spreadsheet", "VOTable", "CSV", "SSV", "IRAF", \
                        "COSV", "TSV", "PSV", "JSON", "CSV_file", "SSV_file", "IRAF_file", \
                        "SSV_file", "TSV_file", "PSV_file", "JSON_file", "WGET_file", "CURL_file"]

        self.base_url = "https://{0}.stsci.edu/"
        self.mission_url = self.base_url.format("archive") + "{0}/{1}/search.php"
        self.other_missions_url = self.base_url.format("archive") + "{0}/search.php"

    def _check_url_exists(self, mission, table):
        if mission in self.missions and table in self.tables_kep or table in self.tables_k2:
            self.url = self.mission_url.format(mission, table)
        elif mission in self.other_missions:
            self.url = self.other_missions_url.format(mission)
        else:
            logger.error("url doesn't exist to make mast request")
            return 1

    def _get_mission_params(self, mission, table=None, params={}, form="JSON", \
                            maxrec=100000, **kwargs):
        if self._check_url_exists(mission, table) == 1:
            return None
        param_dict = {"action": "Search",
                      "showquery": "on",
                      "max_records": maxrec,
                      "outputformat": form,
                      "params": params
        }
        param_dict.update(kwargs)
        r = session(self.url).get(params=param_dict)
        if r.status_code == 200:
            logger.info("sent MAST request")
            return r
        return None

    def _parse_json_output(self, mission, params, output_params, table=None, \
                           form="JSON", maxrec=100000, **kwargs):
        new_arr = []
        r = self._get_mission_params(mission, params, table, form, maxrec, **kwargs)
        if r is None:
            return None
        json_arr = r.json()
        for json_obj in json_arr:
            new_dict = {}
            for key in output_params:
                if key in json_obj:
                    new_dict[key] = json_obj[key]
            new_arr.append(new_dict)
        return new_arr

    def get_target_params(self, params, **kwargs):
        return self._get_mission_params("kepler", "kepler_fov", params, "JSON", **kwargs)

    def get_caom_params(self, columns, filters, form="json", page_size=400000, **kwargs):
        url = self.base_url.format("mast") + "api/v0/invoke"
        columns_str = format_arr(columns, ",")
        json_req = {"service": "Mast.Caom.Filtered",
                    "pagesize": page_size,
                    "format": form,
                    "params": {
                        "columns": columns_str,
                        "filters": filters
                    }
        }
        json_req.update(kwargs)
        json_str = json.dumps(json_req)
        r = session(mast_url).get(params={
            "request": json_str
        })
        if r.status_code == 200:
            logger.info("sent MAST CAOM request")
            return r
        return None
