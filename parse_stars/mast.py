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
    """
    get parameters from mast, see mast for parameters
    """
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

    def _check_url_exists(self, mission, table, form):
        if form not in self.outputs:
            logger.error("format doesn't exist")
            return 1
        if mission in self.missions and table in self.tables_kep or table in self.tables_k2:
            self.url = self.mission_url.format(mission, table)
        elif mission in self.other_missions:
            self.url = self.other_missions_url.format(mission)
        else:
            logger.error("url doesn't exist to make mast request")
            return 1

    def get_mission_params(self, mission, table=None, output_params="", \
                            form="JSON", maxrec=100000, **params):
        """
        gets all parameters given from mast for given missions

        :mission: mission data
        :table: some missions have multiple tables/databases
        :output_params: list-separated string of desired parameters to show
        :form: output format
        :maxrec: maximum records shown
        :params: other paramters to request from mast with

        :return: output formatted string
        """
        if self._check_url_exists(mission, table, form) == 1:
            return None
        param_dict = {"action": "Search",
                      "max_records": maxrec,
                      "outputformat": form,
        }
        param_dict.update(params)
        if output_params != "":
            param_dict.update({"selectedColumnsCsv": output_params})
        r = session(self.url).get(params=param_dict)
        if r.status_code == 200:
            logger.info("sent MAST request")
            return r
        return None

    def parse_json_output(self, mission, table=None, output_params="", \
                           maxrec=100000, **params):
        """
        gets parameters from given mission but properly json-formatted

        :mission, table, output_params, maxrec, params: see above
        """
        new_arr = []
        r = self.get_mission_params(mission, table, form="JSON", maxrec=maxrec, **params)
        if r is None:
            return None
        try:
            json_arr = r.json()
        except Exception as e:
            logger.error("Error: %s" % e.message)
            return None
        if output_params is None:
            new_arr = json_arr
        else:
            for json_obj in json_arr:
                new_dict = {}
                for key in output_params:
                    if key in json_obj:
                        new_dict[key] = json_obj[key]
                new_arr.append(new_dict)
        return new_arr

    def get_caom_params(self, columns, filters, form="json", page_size=400000, **kwargs):
        """
        gets CAOM parameters straight from mast api

        :columns: parameters
        :filters: any filters for results
        :form: output format
        :page_size: maximum records shown
        :kwargs: remaining arguments for api
        """
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
