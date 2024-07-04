"""
A module containing base classes for (pre-)processing the data of Amazon.com reviews.
"""

import calendar
import functools
import gc
import inspect
import multiprocessing
import os
import shutil
import tempfile
import zipfile

import contractions
import natsort
import numpy as np
from pyhelpers._cache import _check_dependency
from pyhelpers.dirs import cd, cdd
from pyhelpers.settings import mpl_preferences
from pyhelpers.store import load_data, save_data, xlsx_to_csv
from pyhelpers.text import find_similar_str, remove_punctuation

from src.processor.utils import *
from src.utils import CustomerReviewsAnalysis, identify_language, remove_digits, \
    remove_single_letters


class _Reviews:
    """
    A class for preprocessing the data of reviews collected from Amazon.com.
    """

    #: Name of the product.
    PRODUCT_NAME: str = 'Product name'
    #: Category of the product.
    PRODUCT_CATEGORY: str | None = None
    #: Type of the product.
    PRODUCT_TYPE: str | None = None

    #: Default column name of original review text.
    ORIGINAL_REVIEW_COLUMN_NAME: str = ''
    #: Default column name of preprocessed review text.
    PROCESSED_REVIEW_COLUMN_NAME: str = 'review_text'
    #: Default column name of `VADER sentiment <https://github.com/cjhutto/vaderSentiment>`_ score.
    VADER_COLUMN_NAME: str = 'vs_compound_score'
    #: Default column name of sentiment label.
    SENTIMENT_COLUMN_NAME: str = 'sentiment'

    #: Schema name.
    SCHEMA_NAME: str = '_reviews'
    #: Table name.
    TABLE_NAME: str = '_product_name'
    #: Full table in PostgreSQL query statement.
    TABLE_IN_QUERY: str = f'"{SCHEMA_NAME}"."{TABLE_NAME}"'
    #: PostgreSQL query statement to read the whole table.
    SQL_QUERY: str = f'SELECT * FROM {TABLE_IN_QUERY}'

    def __init__(self, db_instance=None, update=False, verbose=True,
                 load_preprocd_data=False, load_prep_data=False, load_raw_data=False,
                 verified_reviews_only=False, word_count_threshold=20, dual_scale=False,
                 use_db=False, **kwargs):
        """
        :param db_instance: An instance of the project database. Defaults to ``None``.
        :type db_instance: None | CustomerReviewsAnalysis
        :param update: Whether to reprocess the original data file(s). Defaults to ``False``.
        :type update: bool | int
        :param verbose: Whether to print relevant information in the console. Defaults to ``True``.
        :type verbose: bool | int
        :param load_preprocd_data: Whether to load the preprocessed data. Defaults to ``False``.
        :type load_preprocd_data: bool
        :param load_prep_data: Whether to load the preparatory data. Defaults to ``False``.
        :type load_prep_data: bool
        :param load_raw_data: Whether to load the raw data. Defaults to ``False``.
        :type load_raw_data: bool
        :param verified_reviews_only: Whether to consider only the verified reviews;
            defaults to ``False``.
        :type verified_reviews_only: bool
        :param word_count_threshold: Word count in a review,
            beyond which the review is not considered for further analysis. Defaults to ``20``.
        :type word_count_threshold: int
        :param dual_scale: Whether the sentiment is determined based on both rating and
            VADER sentiment score. Defaults to ``False``.
        :type dual_scale: bool
        :param use_db: Whether to use the database. Defaults to ``False``.
        :type use_db: bool
        :param kwargs: [Optional] Parameters for the methods:
            :meth:`~src.processor._Base.load_prep_data`,
            :meth:`~src.processor._Base.load_preprocd_data` and
            :meth:`~src.processor._Base.load_raw_data`.

        :ivar dict raw_column_name_changes: Changes in column names of the raw data.
        :ivar dict column_name_changes: Changes in column names of the preparatory data.
        :ivar list index_names: Names of the columns used as index when stored in the database.
        :ivar list sentiment_column_names: Names of the columns indicating sentiment.
        :ivar bool verified_reviews_only: Whether to consider only the verified reviews.
        :ivar int word_count_threshold:
            Review word count threshold; reviews longer than this are excluded.
        :ivar bool dual_scale: Whether sentiment analysis uses both rating and VADER scores.
        :ivar bool use_db: Whether the class instance uses the database.
        :ivar None | CustomerReviewsAnalysis db_instance: Instance of the project database.
        :ivar pandas.DataFrame | None raw_data: Raw data loaded from the source.
        :ivar pandas.DataFrame | None prep_data: Data prepared for preprocessing.
        :ivar pandas.DataFrame | None preprocd_data: Preprocessed data with sentiment labels.

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners()
            >>> rvc.PRODUCT_NAME
            'Robotic vacuum cleaners'
            >>> rvc.preprocd_data.shape
            (101608, 19)
            >>> rvc = RoboticVacuumCleaners(verified_reviews_only=True)
            >>> rvc.preprocd_data.shape
            (89988, 19)
            >>> tvc = TraditionalVacuumCleaners()
            >>> tvc.PRODUCT_NAME
            'Traditional vacuum cleaners'
            >>> tvc.preprocd_data.shape
            (146656, 19)
            >>> tvc = TraditionalVacuumCleaners(verified_reviews_only=True)
            >>> tvc.preprocd_data.shape
            (131998, 19)
        """

        self.raw_column_name_changes = {
            'PageUrl': 'ReviewPageURL',
            'Link_Url': 'ProductPageURL',
            'Product_Title': 'ProductTitle',
            'Parent_Id': 'ParentID',
            'Review_Date': 'ReviewDate',
            'Reviewer_Name': 'ReviewerName',
            'Review_Title': 'ReviewTitle',
            'Review_Text': 'ReviewText',
            'People_Found_Helpful': 'PeopleFoundHelpful',
        }

        self.column_name_changes = {
            'ReviewDate': ['review_date', 'review_location'],
            'PeopleFoundHelpful': 'people_found_helpful',
            'Rating': 'rating',
        }

        self.index_names = ['Brand', 'ASIN', 'ParentID', 'review_date']

        self.sentiment_column_names = [
            self.SENTIMENT_COLUMN_NAME + f'_on_{x}' for x in ['rating', 'vs_score', 'dual_scale']]

        self.raw_data, self.prep_data, self.preprocd_data = None, None, None
        self.preprocd_data_ = None

        self.verified_reviews_only = verified_reviews_only
        self.word_count_threshold = word_count_threshold
        self.dual_scale = dual_scale

        self.db_instance = db_instance
        self.use_db = use_db

        load_args = {
            'verified_reviews_only': self.verified_reviews_only,
            'update': update,
            'verbose': verbose,
        }
        kwargs.update(load_args)

        if load_raw_data:
            self.load_raw_data(**kwargs)

        if load_prep_data:
            self.load_prep_data(**kwargs)

        if load_preprocd_data:
            load_args_ = {
                'word_count_threshold': self.word_count_threshold,
                'dual_scale': self.dual_scale,
            }
            self.load_preprocd_data(**(kwargs | load_args_))

    @classmethod
    def cdd(cls, *subdir, mkdir=False, **kwargs):
        """
        Get the full pathname of a directory (or file) under the default data directory.

        :param subdir: name of directory or names of directories (and/or a filename)
        :type subdir: str
        :param mkdir: Whether to create a directory. Defaults to ``False``.
        :type mkdir: bool
        :param kwargs: [Optional] parameters of the function `pyhelpers.dir.cd`_
        :return path: full pathname of a directory (or a file) under ``"data\\"``
        :rtype: str | pathlib.Path

        .. _`pyhelpers.dir.cd`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dirs.cd.html

        **Examples**::

            >>> from src.processor import *
            >>> import os
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> os.path.relpath(rvc.cdd())
            'data\\amazon_reviews\\vacuum_cleaners\\robotic'
            >>> tvc = TraditionalVacuumCleaners(load_preprocd_data=False)
            >>> os.path.relpath(tvc.cdd())
            'data\\amazon_reviews\\vacuum_cleaners\\traditional'
            >>> smt = SmartThermostats(load_preprocd_data=False)
            >>> os.path.relpath(smt.cdd())
            'data\\amazon_reviews\\thermostats\\smart'
        """

        sub_dirs_ = [cls.SCHEMA_NAME, cls.PRODUCT_CATEGORY, cls.PRODUCT_TYPE]
        sub_dirs = [x.lower().replace(' ', '_') for x in sub_dirs_ if isinstance(x, str)]

        sub_dirs += subdir

        path = cdd(*sub_dirs, mkdir=mkdir, **kwargs)

        return path

    @classmethod
    def _get_backup_temp(cls, idx=0):
        backup_zf_pathname = cdd(
            f"_backups\\{cls.SCHEMA_NAME}\\{cls.PRODUCT_CATEGORY.lower().replace(' ', '_')}",
            f"{cls.PRODUCT_TYPE.lower().replace(' ', '_')}.zip")

        with zipfile.ZipFile(backup_zf_pathname, mode='r') as zf:
            if idx < len(zf.filelist):
                f = zf.filelist[idx]
                _, ext = os.path.splitext(f.filename)
                f_ = zf.extract(f.filename, tempfile.tempdir)
            else:
                print("`idx` exceeds the maximum index in the backup archive file.")
                f_ = None

        return f_

    def _read_raw_data(self, path_to_file, verbose=False):
        """
        Read and preprocess the original data of product reviews.

        :param path_to_file: Pathname of the raw data file.
        :type path_to_file: str | pathlib.Path
        :param verbose: Whether to print relevant information in the console. Defaults to ``False``.
        :type verbose: bool | int
        :return: Roughly preprocessed data of the product reviews as a pandas DataFrame,
            or ``None`` if an error occurs.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.processor import *
            >>> import os
            >>> # Robotic vacuum
            >>> rvc = RoboticVacuumCleaners()
            >>> temp_path_to_file = rvc._get_backup_temp(idx=0)
            >>> raw_dat = rvc._read_raw_data(temp_path_to_file)
            >>> raw_dat.shape
            (19946, 13)
            >>> os.remove(temp_path_to_file)
            >>> # Traditional vacuum
            >>> tvc = TraditionalVacuumCleaners()
            >>> temp_path_to_file = tvc._get_backup_temp(idx=0)
            >>> raw_dat = tvc._read_raw_data(temp_path_to_file)
            >>> raw_dat.shape
            (19946, 13)
            >>> os.remove(temp_path_to_file)
            >>> # Smart thermostats
            >>> smt = SmartThermostats()
            >>> temp_path_to_file = smt._get_backup_temp(idx=0)
            >>> raw_dat = smt._read_raw_data(temp_path_to_file)
            >>> raw_dat.shape
            (19998, 13)
            >>> os.remove(temp_path_to_file)
        """

        if path_to_file.endswith(".xlsx"):
            temp_csv_pathname = xlsx_to_csv(path_to_file, sheet_name='1')
            raw_data = pd.read_csv(
                temp_csv_pathname, encoding="ISO-8859-1", keep_default_na=False, low_memory=False)
            os.remove(temp_csv_pathname)  # Remove the temporary CSV file
        else:
            raw_data = pd.read_csv(path_to_file, keep_default_na=False, low_memory=False)

        pic_urls_col = 'Picture_Urls'
        # if raw_data[pic_urls_col].isnull().values.all(): raw_data.drop(pic_urls_col, axis=1)
        if pic_urls_col in raw_data.columns:
            del raw_data[pic_urls_col]  # Drop 'Picture_Urls', which is all NaN or empty string

        # 'Review_Text'
        review_text_col = 'Review_Text'
        void_text1 = 'This is a modal window.' \
                     'No compatible source was found for this media.'
        void_text2 = 'This is a modal window.' \
                     'The media could not be loaded, either because the server or ' \
                     'network failed or because the format is not supported.'
        raw_data.loc[:, review_text_col] = raw_data[review_text_col].map(
            lambda x: x.replace(void_text1, '').replace(void_text2, '').replace(
                '\n\n', ' ').replace('\n', ' ').replace('\xa0', '').replace('\xa1', '!').replace(
                '\xbd', '1/2').replace(' \xe1s ', ' as ').replace('ci\xf3n', 'tion').replace(
                '\x80', '€').replace('\x84', '"').replace('\x85', '...').replace(
                '\x91', "'").replace('\x92', "'").replace('\x93', '"').replace('\x94', '"').replace(
                '\x95', '_').replace('\x96', ' - ').replace('\x97', ' - ').replace(
                '  ', ' ').strip().rstrip(';'))

        # 'Product_Title'
        prod_title_col = 'Product_Title'
        asin_prod_dict = dict(zip(raw_data.ASIN, raw_data[prod_title_col]))
        nan_prod_title = raw_data[raw_data[prod_title_col].isna()]
        if not nan_prod_title.empty:
            for i in nan_prod_title.index:
                asin = raw_data.ASIN.loc[i]
                raw_data.loc[i, prod_title_col] = asin_prod_dict[asin]

        raw_data[prod_title_col] = (
            raw_data[prod_title_col].str.replace(', ', ',').str.replace(',', ', '))

        # Remove duplicates
        subset_cols = ['Parent_Id', 'Review_Date', 'Review_Title', 'Review_Text']
        raw_data.drop_duplicates(subset=subset_cols, keep='first', inplace=True)

        raw_data.rename(columns=self.raw_column_name_changes, inplace=True)

        if verbose:
            print("Done.")

        total_rows, counter = len(raw_data), 1

        if verbose == 3:
            print(f"Total of records: {total_rows}.")

        for col in raw_data.columns:
            tmp = raw_data[pd.isnull(raw_data[col])]

            end_mark = ";" if counter < len(raw_data.columns) else "."

            if len(tmp) != 0 and verbose == 3:
                print(f"\t{len(tmp)} in \"{col}\" are NaN, replaced with ''{end_mark}")

                if raw_data[col].dtype.name == 'object':
                    raw_data[col] = raw_data[col].fillna('')

            counter += 1

        return raw_data

    def read_raw_data(self, path_to_file, verbose=False):
        """
        Read and preprocess the original product review data.

        :param path_to_file: Pathname of the raw data file.
        :type path_to_file: str | pathlib.Path
        :param verbose: Whether to print relevant information in the console. Defaults to ``False``.
        :type verbose: bool | int
        :return: Roughly-preprocessed data of the product reviews.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> from pyhelpers.dirs import cdd
            >>> import os
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> temp_path_to_file = rvc._get_backup_temp(idx=0)
            >>> raw_dat = rvc.read_raw_data(temp_path_to_file, verbose=3)
            Total of records: 92742.
            >>> raw_dat.shape
            (92742, 13)
            >>> os.remove(temp_path_to_file)
        """

        # file_pathnames = glob.glob(rvc.cdd("amazon_reviews", "*.xlsx"))
        # path_to_file = file_pathnames[0]

        filename_ = os.path.splitext(os.path.basename(path_to_file))[0]
        if verbose == 2:
            print(f"\t\"{filename_}\"", end=" ... ")
        elif verbose:
            print(f"Reading \"{filename_}\"", end=" ... ")

        try:
            raw_data = self._read_raw_data(path_to_file=path_to_file, verbose=verbose)
        except Exception as e:
            print(f"Failed. {e}")
            raw_data = None

        return raw_data

    def if_is_verified_note(self):
        """
        Returns a note message indicating whether the data is considered verified only.

        :return: A note message indicating whether the data is verified only.
        :rtype: str

        **Examples**:

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> rvc.verified_reviews_only
            False
            >>> rvc.if_is_verified_note()
            ''
        """

        verified_note = " (verified reviews only)" if self.verified_reviews_only else ""

        return verified_note

    def _prep_raw_data(self, verbose_, index_columns, reset_index=False):
        if verbose_:
            print(f"Reading the raw data of {self.PRODUCT_NAME.lower()} reviews ... ")

        if self.PRODUCT_TYPE is None:
            raw_data = None

        else:
            archive_pathname = cdd(
                f"_backups\\{self.SCHEMA_NAME}\\{self.PRODUCT_CATEGORY.lower().replace(' ', '_')}",
                f"{self.PRODUCT_TYPE.lower().replace(' ', '_')}.zip")

            with zipfile.ZipFile(archive_pathname, mode='r') as archive:
                raw_data_list = []
                for f in natsort.natsorted(archive.filelist, key=lambda x: x.filename):
                    _, ext = os.path.splitext(f.filename)
                    f_ = archive.extract(f.filename, tempfile.tempdir)
                    raw_data_list.append(self.read_raw_data(f_, verbose=verbose_))
                    os.remove(f_)

            raw_data = pd.concat(raw_data_list, axis=0, ignore_index=True)

            subset1 = ['ASIN', 'ParentID']
            raw_data.drop_duplicates(subset=subset1, keep='first', inplace=True)

            subset2 = ['ReviewDate', 'ReviewTitle', 'ReviewText']
            raw_data.drop_duplicates(subset=subset2, keep='first', inplace=True)

            raw_data['PeopleFoundHelpful'] = raw_data['PeopleFoundHelpful'].fillna('')

            if not reset_index:
                raw_data.set_index(index_columns, inplace=True)

        return raw_data

    def load_raw_data(self, index_columns=None, verified_reviews_only=False, update=False,
                      verbose=False, ret_data=False, **kwargs):
        """
        Reads the original version (raw data) of product reviews.

        :param index_columns: Name(s) of column(s) to set as the index;
            defaults to ``['Brand', 'ASIN', 'ParentID']`` if not specified.
        :type index_columns: str | list | None
        :param verified_reviews_only: Whether to consider only verified reviews;
            defaults to ``False``.
        :type verified_reviews_only: bool
        :param update: Whether to reprocess the original data file(s). Defaults to ``False``.
        :type update: bool | int
        :param verbose: Whether to print relevant information in the console. Defaults to ``False``.
        :type verbose: bool | int
        :param ret_data: Whether to return the raw data that is read/loaded. Defaults to ``False``.
        :type ret_data: bool
        :param kwargs: [Optional] parameters for the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`.
        :type kwargs: dict
        :return: Original version (raw data) of the product reviews.
        :rtype: pandas.DataFrame | None

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`: 
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> rvc.load_raw_data(verbose=True)
            >>> rvc.raw_data.shape
            (143217, 13)
        """

        if self.raw_data is None or update:
            if verbose:
                verbose_ = 2
            else:
                verbose_ = False

            if index_columns is None:
                index_columns_ = ['Brand', 'ASIN', 'ParentID']
            else:
                index_columns_ = index_columns

            if self.use_db:
                if self.db_instance is None:
                    self.db_instance = CustomerReviewsAnalysis()

                if self.db_instance.table_exists(self.TABLE_NAME, self.SCHEMA_NAME) and not update:
                    if verbose_:
                        verified_note = self.if_is_verified_note()
                        print(f"Loading the raw data of the {self.PRODUCT_NAME.lower()} reviews" +
                              verified_note, end=" ... ")

                    self.raw_data = self.db_instance.read_sql_query(
                        sql_query=self.SQL_QUERY, true_values=['t', 'true'],
                        false_values=['f', 'false'], keep_default_na=False, **kwargs).sort_values(
                        index_columns_, ignore_index=True)

                    if verbose_:
                        print("Done.")

                else:
                    self.raw_data = self._prep_raw_data(
                        verbose_=verbose_, index_columns=index_columns_)

                    self.db_instance.import_data(
                        data=self.raw_data, table_name=self.TABLE_NAME,
                        schema_name=self.SCHEMA_NAME, index=True, if_exists='replace',
                        method=self.db_instance.psql_insert_copy, confirmation_required=False,
                        verbose=verbose_, **kwargs)

                    self.db_instance.null_text_to_empty_string(
                        table_name=self.TABLE_NAME, schema_name=self.SCHEMA_NAME)

                    self.raw_data.reset_index(inplace=True)

            else:
                path_to_pkl = self.cdd("raw_data", "raw_data.pkl")

                if not os.path.isfile(path_to_pkl) or update:
                    self.raw_data = self._prep_raw_data(
                        verbose_=verbose_, index_columns=index_columns_, reset_index=True)

                    save_data(self.raw_data, path_to_pkl, verbose=verbose)

                else:
                    self.raw_data = load_data(path_to_pkl, verbose=verbose)

        else:
            if verbose:
                verified_note = self.if_is_verified_note()
                print(f"The raw data{verified_note} is already loaded.")

        self.verified_reviews_only = verified_reviews_only

        if self.verified_reviews_only:
            self.raw_data.query('`Verified` == True', inplace=True)

        if ret_data:
            return self.raw_data

    # == Prepare the data for preprocessing ========================================================

    def _check_prep_data(self, refresh=False, verbose=False):
        """
        Checks whether the preparatory data is already loaded or needs to be updated.
        """

        if self.prep_data is None or refresh:
            if self.raw_data is None:
                self.load_raw_data(verbose=verbose)

            self.prep_data = self.raw_data.copy()

    def convert_to_integer(self, column_names, int_type=np.uint8, refresh=False):
        """
        Convert float values to integers.

        :param column_names: Name of a column or list of column names to convert.
        :type column_names: str | list
        :param int_type: Specific integer type to cast the values. Defaults to ``np.uint8``.
        :type int_type: type
        :param refresh: Whether to apply the conversion on the raw or preparatory data and
            update the preprocessed data. Defaults to ``False``.
        :type refresh: bool

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> rating_col_name = 'Rating'
            >>> rvc.raw_data[rating_col_name] = rvc.raw_data[rating_col_name].astype(float)
            >>> rvc.raw_data[rating_col_name].head()
            0    2.0
            1    3.0
            2    1.0
            3    1.0
            4    2.0
            Name: Rating, dtype: float64
            >>> rvc.convert_to_integer(column_names=rating_col_name)
            >>> rvc.prep_data['rating'].head()
            0    2
            1    3
            2    1
            3    1
            4    2
            Name: rating, dtype: uint8
        """

        self._check_prep_data(refresh=refresh)

        if isinstance(column_names, str):
            column_names = [column_names]
        column_names = [x for x in column_names if x in self.prep_data.columns]

        if len(column_names) > 0:
            column_names_ = [x.replace(' ', '_').lower() for x in column_names]
            column_name_changes = dict(zip(column_names, column_names_))

            self.prep_data[column_names] = self.prep_data[column_names].astype(int_type).values

            self.prep_data.rename(columns=column_name_changes, inplace=True)

            self.column_name_changes.update(column_name_changes)

    @staticmethod
    def _parse_review_date(s):
        return s.map(parse_review_date)

    def parse_review_date(self, column_name='ReviewDate', parsed_column_name=None, refresh=False):
        """
        Parse the information about dates for each record of the product reviews.

        :param column_name: Name of the column that contains information about review dates;
            defaults to ``'ReviewDate'``.
        :type column_name: str
        :param parsed_column_name: New column names for parsed date data,
            in cases where original records contain both date and location information;
            defaults to ``None``.
        :type parsed_column_name: list | None
        :param refresh: Whether to perform this function on the raw data. Defaults to ``False``.
        :type refresh: bool

        .. note::

            Newly created column names are set to lowercase by default.

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> rvc.raw_data['ReviewDate'].head()
            0    July 12, 2021
            1     July 7, 2021
            2    July 11, 2021
            3     July 9, 2021
            4    July 12, 2021
            Name: ReviewDate, dtype: object
            >>> rvc.parse_review_date()  # Transform the review date data
            >>> new_column_names = rvc.column_name_changes['ReviewDate']
            >>> new_column_names
            ['review_date', 'review_location']
            >>> rvc.prep_data[new_column_names].head()
              review_date review_location
            0  2021-07-12   United States
            1  2021-07-07   United States
            2  2021-07-11   United States
            3  2021-07-09   United States
            4  2021-07-12   United States
        """

        self._check_prep_data(refresh=refresh)

        if column_name in self.prep_data.columns:
            n_processes = os.cpu_count() - 1
            dat_list = np.array_split(self.prep_data[column_name], n_processes)
            with multiprocessing.Pool(processes=n_processes) as p:
                date_and_loc = pd.concat(p.map(self._parse_review_date, dat_list))

            if parsed_column_name is None:
                column_name_ = ['review_date', 'review_location']
            else:
                column_name_ = parsed_column_name.copy()

            parsed_date_and_loc = pd.DataFrame(
                data=date_and_loc.to_list(), columns=column_name_, index=self.prep_data.index)

            del self.prep_data[column_name]
            gc.collect()

            self.prep_data = pd.concat([self.prep_data, parsed_date_and_loc], axis=1)

            self.column_name_changes.update({column_name: column_name_})

    def regulate_people_found_helpful(self, column_name='PeopleFoundHelpful', refresh=False):
        """
        Regulates the data regarding how many people found reviews helpful.

        :param column_name: Name of the column that contains information about the number of people
            who found a review helpful. Defaults to ``'PeopleFoundHelpful'``.
        :type column_name: str
        :param refresh: Whether to perform this function on the raw data and
            update the preparatory data. Defaults to ``False``.
        :type refresh: bool

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> rvc.raw_data['PeopleFoundHelpful'].head()
            0
            1    2
            2
            3
            4
            Name: PeopleFoundHelpful, dtype: object
            >>> rvc.regulate_people_found_helpful()  # Cleanse the data
            >>> new_column_name = rvc.column_name_changes['PeopleFoundHelpful']
            >>> new_column_name
            'people_found_helpful'
            >>> rvc.prep_data[new_column_name].head()
            0    0
            1    2
            2    0
            3    0
            4    0
            Name: people_found_helpful, dtype: int64
        """

        self._check_prep_data(refresh=refresh)

        if column_name in self.prep_data.columns:
            temp = self.prep_data[column_name].map(regulate_people_found_helpful)

            column_name_ = 'people_found_helpful'
            column_name_change = {column_name: column_name_}

            self.prep_data.rename(columns=column_name_change, inplace=True)

            # self.prep_data.loc[:, column_name_] = temp
            i = self.prep_data.columns.get_loc(column_name_)
            self.prep_data[self.prep_data.columns[i]] = temp.values

            del temp
            gc.collect()

            self.column_name_changes.update(column_name_change)

    @classmethod
    def specify_sql_query(cls, table_name, before_date=None, verified_reviews_only=False):
        """
        Specify SQL statement for querying data.

        :param table_name: Name of the table to query.
        :type table_name: str
        :param before_date: Filter data to include only records before this date (exclusive).
            Defaults to ``None``.
        :type before_date: str | None
        :param verified_reviews_only: Whether to include only verified reviews in the query;
            Defaults to ``True``.
        :type verified_reviews_only: bool
        :return: SQL query string.
        :rtype: str

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> rvc.specify_sql_query(table_name='<table_name>')
            'SELECT * FROM amazon_reviews."<table_name>"'
            >>> rvc.specify_sql_query(table_name='<table_name>', before_date='2022-01-01')
            'SELECT * FROM amazon_reviews."<table_name>" WHERE "review_date" < \'2022-01-01\''
            >>> rvc.specify_sql_query(table_name='<table_name>', verified_reviews_only=True)
            'SELECT * FROM amazon_reviews."<table_name>" WHERE "Verified" IS TRUE'
        """

        sql_query = f'SELECT * FROM {cls.SCHEMA_NAME}."{table_name}"'

        if before_date:
            sql_query += f' WHERE "review_date" < \'{before_date}\''

        if verified_reviews_only:
            pref = ' WHERE ' if 'WHERE' not in sql_query else ''
            sql_query += pref + '"Verified" IS TRUE'

        return sql_query

    def _filter_data(self, name, before_date=None, verified_reviews_only=False):
        """
        Filter data based on specified conditions for the given column ``name``.
        """

        if before_date:
            getattr(self, name).query(f"`review_date` < '{before_date}'", inplace=True)

        if verified_reviews_only:
            getattr(self, name).query('`Verified` == True', inplace=True)

    @classmethod
    def correct_identified_typos(cls, review_text, processes=None):
        """
        Correct typos that have been identified.

        :param review_text: textual data of product reviews
        :type review_text: pandas.Series | list
        :param processes: number of worker processes to use by `multiprocessing.Pool()`_,
            defaults to ``None``
        :type processes: int
        :return: textual data of which identified typos are corrected
        :rtype: pandas.Series

        .. _`multiprocessing.Pool()`:
            https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> sample = rvc.raw_data['ReviewText'].sample(random_state=3)
            >>> sample
            63709    Great investment, must have got every carpet h...
            Name: ReviewText, dtype: object
            >>> sample_ = rvc.correct_identified_typos(sample)
            >>> sample_
            63709    Great investment, must have got every carpet h...
            dtype: object
            >>> sample.equals(sample_)
            False
        """

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore')

        n_processors = os.cpu_count() - 1 if processes is None else processes
        with multiprocessing.Pool(processes=n_processors) as p:
            rev_txt_ = p.map(functools.partial(correct_identified_typos, split=False), review_text)
            rev_txt = p.map(contractions.fix, rev_txt_)

        review_text_ = pd.Series(rev_txt, index=review_text.index)

        return review_text_

    def make_prep_data(self, ret_prep_data=False, verbose=False):
        """
        Make preparatory data from the raw data.

        :param ret_prep_data: Whether to return the preparatory data. Defaults to ``False``.
        :type ret_prep_data: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int
        :return: the preparatory data (when ``ret_prep_data=True``)
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> rvc.prep_data is None
            True
            >>> rvc.make_prep_data(verbose=True)
            >>> rvc.prep_data.head()
                     ASIN   Brand  ... review_date review_location
            0  B085D45SZF  iRobot  ...  2021-07-12   United States
            1  B08TN2GC94  iRobot  ...  2021-07-07   United States
            2  B085D45SZF  iRobot  ...  2021-07-11   United States
            3  B085D45SZF  iRobot  ...  2021-07-09   United States
            4  B085D45SZF  iRobot  ...  2021-07-12   United States
            [5 rows x 12 columns]
            >>> rvc.prep_data.shape
            (143217, 12)
            >>> tvc = TraditionalVacuumCleaners(load_raw_data=True, load_preprocd_data=False)
            >>> tvc.prep_data is None
            True
            >>> tvc.make_prep_data(verbose=True)
            >>> tvc.prep_data.head()
                     ASIN   Brand  ... review_date review_location
            0  B07SMJJTL7  EUREKA  ...  2022-09-19   United States
            1  B07SMJJTL7  EUREKA  ...  2022-09-19   United States
            2  B07SMJJTL7  EUREKA  ...  2022-09-19   United States
            3  B07SMJJTL7  EUREKA  ...  2022-09-19   United States
            4  B07SMJJTL7  EUREKA  ...  2022-09-18   United States
            [5 rows x 12 columns]
            >>> tvc.prep_data.shape
            (230479, 12)
        """

        if self.raw_data is None:
            self.load_raw_data()

        if verbose is True:
            print("Making data ready for preprocessing", end=" ... ")

        try:
            self.prep_data = self.raw_data.drop(['ReviewPageURL', 'ProductPageURL'], axis=1)

            self.convert_to_integer(column_names='Rating')

            self.parse_review_date()

            if 'traditional' in self.SCHEMA_NAME.lower():
                self.prep_data = self.prep_data.query('`review_date` >= "2011-01-01"')

            self.regulate_people_found_helpful()

            review_text = self.prep_data[self.ORIGINAL_REVIEW_COLUMN_NAME]

            # Correct potential typos and known typos that have been found
            review_text_ = self.correct_identified_typos(review_text)

            self.prep_data.loc[:, self.ORIGINAL_REVIEW_COLUMN_NAME] = review_text_

            if verbose:
                print("Done.")

        except Exception as e:
            print(f"Failed. {e}")

        if ret_prep_data:
            return self.prep_data

    def load_prep_data(self, before_date=None, verified_reviews_only=False, update=False,
                       verbose=False, ret_data=False, **kwargs):
        """
        Load the preparatory version of the product reviews data.

        :param before_date: date before which the preparatory data is considered,
            e.g. '2021-04-01' and '2022-04-01'. Defaults to ``None``.
        :type before_date: str | None
        :param verified_reviews_only: consider only the verified reviews. Defaults to ``False``.
        :type verified_reviews_only: bool
        :param update: Whether to reprocess the original data file(s). Defaults to ``False``.
        :type update: bool | int
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int
        :param ret_data: Whether to return the raw data that is read/loaded. Defaults to ``False``.
        :type ret_data: bool
        :param kwargs: [Optional] parameters of the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query`_.
        :return: Preparatory data of product reviews
        :rtype: pandas.DataFrame | None

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> # rvc.load_prep_data(update=True, verbose=True)  # Update prep_data
            >>> rvc.load_prep_data(verbose=True)
            >>> rvc.prep_data.shape
            (143217, 12)
            >>> tvc = TraditionalVacuumCleaners(load_preprocd_data=False)
            >>> # tvc.load_prep_data(update=True, verbose=True)  # Update prep_data
            >>> tvc.load_prep_data(verbose=True)
            >>> tvc.prep_data.shape
            (230479, 12)
        """

        if self.verified_reviews_only != verified_reviews_only:
            self.preprocd_data = None
            self.verified_reviews_only = verified_reviews_only

        if self.prep_data is None or update:
            if self.use_db:
                if self.db_instance is None:
                    self.db_instance = CustomerReviewsAnalysis()

                table_name = self.TABLE_NAME + '_prep'

                if self.db_instance.table_exists(table_name, self.SCHEMA_NAME) and not update:
                    if verbose:
                        verified_note = self.if_is_verified_note()
                        prt_msg = (f"Loading the preparatory data of the {self.PRODUCT_NAME.lower()} "
                                   f"reviews") + verified_note
                        print(prt_msg, end=" ... ")

                    sql_query = self.specify_sql_query(
                        table_name=table_name, before_date=before_date,
                        verified_reviews_only=self.verified_reviews_only)

                    self.prep_data = self.db_instance.read_sql_query(
                        sql_query=sql_query, parse_dates=['review_date'], keep_default_na=False,
                        true_values=['t', 'true'], false_values=['f', 'false'],
                        **kwargs).sort_values(self.index_names, ignore_index=True)

                    if verbose:
                        print("Done.")

                else:
                    if self.raw_data is None:
                        self.load_raw_data(verbose=verbose)

                    self.make_prep_data(verbose=verbose)
                    assert isinstance(self.prep_data, pd.DataFrame)

                    try:
                        prep_data = self.prep_data.set_index(self.index_names).sort_index()

                        self.db_instance.import_data(
                            data=prep_data, table_name=table_name, schema_name=self.SCHEMA_NAME,
                            index=True, if_exists='replace', confirmation_required=False,
                            method=self.db_instance.psql_insert_copy,
                            verbose=2 if verbose else False)

                        self.db_instance.null_text_to_empty_string(
                            table_name=table_name, schema_name=self.SCHEMA_NAME)

                        self.prep_data = prep_data.reset_index().sort_values(self.index_names)

                    except Exception as e:
                        if verbose:
                            print("Failed. {}".format(e))

            else:
                path_to_pkl = self.cdd("prep_data", "prep_data.pkl")

                if not os.path.isfile(path_to_pkl) or update:
                    self.make_prep_data(verbose=verbose)
                    assert isinstance(self.prep_data, pd.DataFrame)
                    save_data(self.prep_data, path_to_pkl, verbose=verbose)

                else:
                    self.prep_data = load_data(path_to_pkl, verbose=verbose)

        else:
            if verbose:
                verified_note = self.if_is_verified_note()
                print(f"The preparatory data{verified_note} is already loaded.")

        self.prep_data.sort_values(self.index_names, ignore_index=True, inplace=True)

        self._filter_data('prep_data', before_date, self.verified_reviews_only)

        if ret_data:
            return self.prep_data

    # == Descriptive statistics ====================================================================

    @classmethod
    def get_ratings_stats(cls, data, group_label, rating_scores=None, as_percentage=True):
        """
        Calculate proportions of different ratings by year or month.

        :param data: data of the product reviews
        :type data: pandas.DataFrame
        :param group_label: labels used for grouping the data of ratings
        :type group_label: pandas.Series
        :param rating_scores: rating scores that are under investigation. Defaults to ``None``.
        :type rating_scores: int | float | list | None
        :param as_percentage: Whether to return percentages (instead of counts);
            defaults to ``True``.
        :type as_percentage: bool
        :return: proportions of different ratings by year or month (depending on ``group_label``)
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> group_label_yearly = rvc.prep_data.review_date.dt.year
            >>> rvc.get_ratings_stats(rvc.prep_data, group_label_yearly, rating_scores=[5, 1])
                         HighestRating  LowestRating
            review_date
            2013              0.527778      0.069444
            2014              0.553936      0.128280
            2015              0.638298      0.110638
            2016              0.617323      0.077165
            2017              0.608696      0.094629
            2018              0.598578      0.109311
            2019              0.615628      0.112293
            2020              0.628718      0.108077
            2021              0.579268      0.154740
            2022              0.493821      0.210421
            >>> group_label_monthly = rvc.prep_data.review_date.dt.month
            >>> rvc.get_ratings_stats(rvc.prep_data, group_label_monthly, rating_scores=[1, 2])
                         LowestRating  Rating=2*
            review_date
            1                0.124655   0.065256
            2                0.161133   0.073092
            3                0.153101   0.071783
            4                0.139471   0.067167
            5                0.115385   0.062359
            6                0.113606   0.057301
            7                0.109494   0.054627
            8                0.128023   0.060028
            9                0.131382   0.060950
            10               0.135221   0.068209
            11               0.159184   0.077726
            12               0.143404   0.069023
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> group_label_yearly = tvc.prep_data.review_date.dt.year
            >>> tvc.get_ratings_stats(tvc.prep_data, group_label_yearly, rating_scores=[5, 1])
                         HighestRating  LowestRating
            review_date
            2010              0.000000      1.000000
            2011              0.605263      0.131579
            2012              0.511111      0.133333
            2013              0.510870      0.103261
            2014              0.635526      0.068421
            2015              0.694338      0.061591
            2016              0.654757      0.083708
            2017              0.592221      0.136966
            2018              0.594616      0.148855
            2019              0.617327      0.143236
            2020              0.609984      0.146109
            2021              0.580066      0.166213
            2022              0.523127      0.207099
            >>> group_label_monthly = tvc.prep_data.review_date.dt.month
            >>> tvc.get_ratings_stats(tvc.prep_data, group_label_monthly, rating_scores=[1, 2])
                         LowestRating  Rating=2*
            review_date
            1                0.144322   0.062398
            2                0.167498   0.066615
            3                0.168365   0.066849
            4                0.173359   0.065966
            5                0.166426   0.066382
            6                0.165598   0.064445
            7                0.160810   0.066079
            8                0.170584   0.067393
            9                0.172218   0.069717
            10               0.162598   0.062955
            11               0.164632   0.071131
            12               0.152489   0.064626
        """

        assert 'review_date' in data.columns

        col_name = find_similar_str('rating', data.columns)
        data_ = data[col_name]

        if rating_scores is None:
            scores = range(1, 6)
        elif isinstance(rating_scores, (int, float)):
            scores = [rating_scores]
        else:
            scores = list(rating_scores)

        new_names = {}
        for score in scores:
            if score == 1.:
                new_names.update({score: 'LowestRating'})
            elif score == 5.:
                new_names.update({score: 'HighestRating'})
            elif score == 3.:
                new_names.update({score: 'Neutral'})
            else:
                new_names.update({score: 'Rating=%d*' % score})

        ratings_counts = []
        for score in scores:
            ratings_ = pd.concat([data['review_date'], data_.mask(data_.ne(score))], axis=1)
            ratings_counts_ = ratings_.groupby(group_label)[col_name].count()
            ratings_counts_.rename(new_names[score], inplace=True)
            ratings_counts.append(ratings_counts_)
        ratings_counts = pd.concat(ratings_counts, axis=1)

        if as_percentage:
            ratings_all_counts = data.groupby(group_label)[col_name].count()
            ratings_counts = ratings_counts.divide(ratings_all_counts, axis='index')
            # ratings_counts = ratings_counts / len(data)

        return ratings_counts

    def get_descriptive_stats(self, data=None, by='year', before_date='2022-01-01',
                              rating_scores=None, as_percentage=True):
        """
        Get some descriptive statistics.

        :param data: data of the product reviews
        :type data: pandas.DataFrame
        :param by: name of a label by which the descriptive statistics is calculated
        :type by: str
        :param before_date: date before which the data will be considered. Defaults to ``None``.
        :type before_date: datetime.datetime | pandas.Timestamp | str | None
        :param rating_scores: rating scores that are under investigation. Defaults to ``None``.
        :type rating_scores: int | float | list | None
        :param as_percentage: Whether to return percentages (instead of counts);
            defaults to ``True``.
        :type as_percentage: bool
        :return: data of descriptive statistics (proportions)
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # len(rvc.prep_data.Brand.str.lower().unique())  # 81
            >>> # len(rvc.prep_data.ProductTitle.str.lower().unique())  # 284
            >>> # len(rvc.prep_data)  # 143217
            >>> rvc.get_descriptive_stats(by='year')
                            Brand   Product    Review  ...   Neutral  Rating=4*  HighestRating
            review_date                                ...
            2013         0.012346  0.004149  0.000503  ...  0.111111   0.236111       0.527778
            2014         0.012346  0.008299  0.002395  ...  0.075802   0.198251       0.553936
            2015         0.037037  0.020747  0.003282  ...  0.059574   0.136170       0.638298
            2016         0.049383  0.037344  0.008868  ...  0.068504   0.175591       0.617323
            2017         0.086420  0.062241  0.019111  ...  0.070150   0.150895       0.608696
            2018         0.111111  0.099585  0.053017  ...  0.072962   0.148294       0.598578
            2019         0.234568  0.319502  0.121438  ...  0.068537   0.142537       0.615628
            2020         0.493827  0.560166  0.323223  ...  0.066104   0.139811       0.628718
            2021         0.876543  0.879668  0.384542  ...  0.074447   0.122147       0.579268
            2022         0.876543  0.863071  0.083621  ...  0.089178   0.121576       0.493821
            [10 rows x 8 columns]
            >>> rvc.get_descriptive_stats(by='month')
                            Brand   Product    Review  ...   Neutral  Rating=4*  HighestRating
            review_date                                ...
            1            0.753086  0.771784  0.126368  ...  0.078683   0.144988       0.586418
            2            0.802469  0.838174  0.088746  ...  0.079701   0.134461       0.551613
            3            0.827160  0.842324  0.088614  ...  0.074541   0.124813       0.575762
            4            0.876543  0.817427  0.070690  ...  0.068056   0.127222       0.598084
            5            0.530864  0.609959  0.068079  ...  0.065231   0.133333       0.623692
            6            0.666667  0.684647  0.077135  ...  0.060016   0.127274       0.641803
            7            0.691358  0.721992  0.087301  ...  0.067104   0.132528       0.636247
            8            0.691358  0.717842  0.073630  ...  0.067520   0.130109       0.614320
            9            0.703704  0.734440  0.067017  ...  0.066472   0.130756       0.610440
            10           0.728395  0.759336  0.070020  ...  0.074791   0.134324       0.587455
            11           0.765432  0.763485  0.072945  ...  0.077534   0.125586       0.559969
            12           0.777778  0.804979  0.109456  ...  0.076231   0.139768       0.571574
            [12 rows x 8 columns]
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # len(tvc.prep_data.Brand.str.lower().unique())  # 65
            >>> # len(tvc.prep_data.ProductTitle.str.lower().unique())  # 205
            >>> # len(tvc.prep_data)  # 230479
            >>> tvc.get_descriptive_stats(by='year')
                            Brand   Product    Review  ...   Neutral  Rating=4*  HighestRating
            review_date                                ...
            2010         0.015152  0.004926  0.000004  ...  0.000000   0.000000       0.000000
            2011         0.030303  0.009852  0.000165  ...  0.000000   0.236842       0.605263
            2012         0.030303  0.009852  0.000390  ...  0.055556   0.211111       0.511111
            2013         0.075758  0.029557  0.000798  ...  0.076087   0.277174       0.510870
            2014         0.090909  0.044335  0.003297  ...  0.057895   0.197368       0.635526
            2015         0.121212  0.064039  0.011342  ...  0.059679   0.151109       0.694338
            2016         0.136364  0.093596  0.021251  ...  0.065741   0.152103       0.654757
            2017         0.196970  0.152709  0.044507  ...  0.069799   0.135796       0.592221
            2018         0.212121  0.216749  0.070275  ...  0.071248   0.120948       0.594616
            2019         0.287879  0.374384  0.116136  ...  0.061979   0.122053       0.617327
            2020         0.439394  0.556650  0.180875  ...  0.067310   0.116940       0.609984
            2021         0.696970  0.793103  0.308939  ...  0.072145   0.115878       0.580066
            2022         1.000000  1.000000  0.242018  ...  0.081821   0.106508       0.523127
            [13 rows x 8 columns]
            >>> tvc.get_descriptive_stats(by='month')
                            Brand   Product    Review  ...   Neutral  Rating=4*  HighestRating
            review_date                                ...
            1            0.757576  0.807882  0.093106  ...  0.071858   0.121161       0.600261
            2            0.787879  0.822660  0.079135  ...  0.074730   0.116344       0.574812
            3            0.863636  0.881773  0.088984  ...  0.072456   0.117558       0.574772
            4            0.984848  0.940887  0.085045  ...  0.069639   0.114739       0.576297
            5            0.969697  0.950739  0.087257  ...  0.072199   0.111829       0.583163
            6            1.000000  0.975369  0.083214  ...  0.071224   0.112415       0.586318
            7            0.969697  0.931034  0.094946  ...  0.072431   0.117169       0.583512
            8            0.984848  0.950739  0.095991  ...  0.071913   0.117565       0.572546
            9            0.954545  0.916256  0.082585  ...  0.077178   0.113324       0.567563
            10           0.727273  0.798030  0.060786  ...  0.069094   0.123697       0.581656
            11           0.621212  0.724138  0.063620  ...  0.069836   0.122349       0.572052
            12           0.696970  0.783251  0.085331  ...  0.069456   0.122998       0.590431
            [12 rows x 8 columns]
        """

        assert by in {'year', 'month'}, "`by` must be one of {'year', 'month'}"

        if data is None:
            if self.prep_data is None:
                self.load_prep_data(verbose=True)
            dat = self.prep_data.copy()
        else:
            assert 'review_date' in data.columns
            dat = data.copy()

        # if self.PRODUCT_TYPE == 'Traditional':
        dat = dat[dat.review_date >= '2013-01-01']

        if before_date:
            if isinstance(before_date, str):
                before_date_ = pd.to_datetime(before_date)
            else:
                before_date_ = before_date
            dat = dat[dat.review_date < before_date_]

        if by == 'month':
            group_label = dat.review_date.dt.month
        else:  # by == 'year'
            group_label = dat.review_date.dt.year

        # Brands and products
        b_p_col_names = ['Brand', 'ASIN']
        b_p_counts = dat.groupby(group_label)[b_p_col_names].nunique()
        b_p_percents = b_p_counts / dat[b_p_col_names].nunique()
        b_p_percents.rename(columns={'ASIN': 'Product'}, inplace=True)

        # Reviews
        review_counts = dat.reset_index().groupby(group_label)['index'].nunique()
        review_counts.rename('Review', inplace=True)
        review_percents = review_counts / len(dat)

        # Ratings
        ratings_stats = self.get_ratings_stats(
            data=dat, group_label=group_label, rating_scores=rating_scores,
            as_percentage=as_percentage)

        descriptive_stats = pd.concat([b_p_percents, review_percents, ratings_stats], axis=1)

        return descriptive_stats

    def view_stats_on_products(self, data=None, by='year', horizontal=False, save_as=None,
                               verbose=False, **kwargs):
        """
        Make a bar chart of descriptive statistics on the products (and brands).

        :param data: data of the product reviews
        :type data: pandas.DataFrame
        :param by: label by which the descriptive statistics is calculated. Defaults to ``'year'``
        :type by: str
        :param horizontal: Whether to create a horizontal bar chart. Defaults to ``False``.
        :type horizontal: bool
        :param save_as: extension of figure filename, or whether to save the figure;
            defaults to ``None``.
        :type save_as: str | bool | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # rvc.view_stats_on_products(by='year', save_as=".svg", verbose=True)
            >>> rvc.view_stats_on_products(by='year')

        .. figure:: ../_images/robotic_vacuum_cleaners/stats/products_yearly.*
            :name: robotic_vacuum_cleaners_products_yearly
            :align: center
            :width: 80%

            Descriptive statistics of robot vacuums purchased (on a yearly basis).

        .. code-block:: python

            >>> # rvc.view_stats_on_products(by='month', save_as=".svg", verbose=True)
            >>> rvc.view_stats_on_products(by='month')

        .. figure:: ../_images/robotic_vacuum_cleaners/stats/products_monthly.*
            :name: robotic_vacuum_cleaners_products_monthly
            :align: center
            :width: 80%

            Descriptive statistics of robot vacuums purchased (on a monthly basis).

        .. code-block:: python

            >>> from src.processor import TraditionalVacuumCleaners
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # tvc.view_stats_on_products(by='year', save_as=".svg", verbose=True)
            >>> tvc.view_stats_on_products(by='year')

        .. figure:: ../_images/traditional_vacuum_cleaners/stats/products_yearly.*
            :name: traditional_vacuum_cleaners_products_yearly
            :align: center
            :width: 80%

            Descriptive statistics of traditional vacuums purchased (on a yearly basis).

        .. code-block:: python

            >>> # tvc.view_stats_on_products(by='month', save_as=".svg", verbose=True)
            >>> tvc.view_stats_on_products(by='month')

        .. figure:: ../_images/traditional_vacuum_cleaners/stats/products_monthly.*
            :name: traditional_vacuum_cleaners_products_monthly
            :align: center
            :width: 80%

            Descriptive statistics of traditional vacuums purchased (on a monthly basis).

        .. code-block:: python

            >>> from src.processor import SmartThermostats
            >>> smt = SmartThermostats(load_prep_data=True, load_preprocd_data=False)
            >>> # smt.view_stats_on_products(by='year', save_as=".svg", verbose=True)
            >>> smt.view_stats_on_products(by='year')

        .. figure:: ../_images/smart_thermostats/stats/products_yearly.*
            :name: smart_thermostats_products_yearly
            :align: center
            :width: 80%

            Descriptive statistics of smart thermostats purchased (on a yearly basis).

        .. code-block:: python

            >>> # smt.view_stats_on_products(by='month', save_as=".svg", verbose=True)
            >>> smt.view_stats_on_products(by='month')

        .. figure:: ../_images/smart_thermostats/stats/products_monthly.*
            :name: smart_thermostats_products_monthly
            :align: center
            :width: 80%

            Descriptive statistics of smart thermostats purchased (on a monthly basis).
        """

        mpl_preferences(font_size=12, backend='TkAgg')

        plt = _check_dependency(name='matplotlib.pyplot')

        dat = self.get_descriptive_stats(data=data, by=by)

        legend_labels = ['Brand', 'Product']
        brands, products = [dat[label].values for label in legend_labels]

        b_colour = '#F2A154'
        p_colour = '#314E52'

        fig1 = plt.figure(figsize=(6, 4), constrained_layout=True)
        ax1 = fig1.add_subplot()

        bar_width, x_ticks = 0.3, np.arange(len(dat.index))
        tick_labels = list(calendar.month_abbr)[1:] if by == 'month' else dat.index.tolist()

        if horizontal:
            b1 = ax1.barh(x_ticks, brands, height=bar_width, color=b_colour)
            b2 = ax1.barh(x_ticks + bar_width, products, color=p_colour, height=bar_width)

            handles = [b2, b1]
            legend_labels = list(reversed(legend_labels))

            ax1.set_xticks(np.arange(0., 1.1, .2))  # ax1.get_xticks().tolist()[:-1]
            ax1.set_xticklabels(['{:.0%}'.format(x) for x in ax1.get_xticks()])
            ax1.set_yticks(x_ticks + bar_width / 2, tick_labels)

            ax1.set_ylabel(by.title(), fontweight='black', color='#4A403A')

            ax1.set_xlim(xmin=0, xmax=1)

        else:
            b1 = ax1.bar(x_ticks, brands, width=bar_width, color=b_colour)
            b2 = ax1.bar(x_ticks + bar_width, products, color=p_colour, width=bar_width)

            handles = [b1, b2]
            ax1.set_xticks(x_ticks + bar_width / 2, tick_labels)
            ax1.set_yticks(np.linspace(0, 1, 6))  # ax1.get_yticks().tolist()[:-1]
            ax1.set_yticklabels(['{:.0%}'.format(x) for x in ax1.get_yticks()])

            ax1.set_xlabel(by.title(), fontweight='black', color='#4A403A')

            ax1.set_ylim(ymin=0, ymax=1)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax1.spines['left'].set_position(('outward', 8))
        ax1.spines['bottom'].set_position(('outward', 5))

        ax1.legend(
            handles, legend_labels, loc='best',  # bbox_to_anchor=(0.4, 0.8), mode='expand',
            borderaxespad=0, ncol=1, handletextpad=0.1, frameon=False)

        # fig1.tight_layout()

        fig1.show()

        if save_as:
            for save_as_ in {save_as, ".svg", ".pdf"}:
                fig_filename = f"products_{by}ly" + ("_h" if horizontal else "") + save_as_
                path_to_fig = self.cdd("stats", fig_filename, mkdir=True)
                save_data(fig1, path_to_fig, verbose=verbose, **kwargs)

                prod_name = self.PRODUCT_NAME.lower().replace(' ', '_')
                _file_path = f"{prod_name}\\stats\\{fig_filename}"
                docs_file_path = cd(f"docs\\source\\_images\\{_file_path}", mkdir=True)
                shutil.copyfile(path_to_fig, docs_file_path)

    def view_stats_on_ratings(self, data=None, by='year', review_stats=True, horizontal=False,
                              save_as=None, verbose=False, **kwargs):
        """
        Create a bar chart of descriptive statistics on customers' ratings
        (and proportions of reviews).

        :param data: data of the product reviews
        :type data: pandas.DataFrame
        :param by: label by which the descriptive statistics is calculated. Defaults to ``'year'``
        :type by: str
        :param review_stats: Whether to include the proportions of reviews. Defaults to ``True``.
        :type review_stats: bool
        :param horizontal: Whether to create a horizontal bar chart. Defaults to ``False``.
        :type horizontal: bool
        :param save_as: extension of figure filename, or whether to save the figure;
            defaults to ``None``.
        :type save_as: str | bool | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # rvc.view_stats_on_ratings(by='year', save_as=".svg", verbose=True)
            >>> rvc.view_stats_on_ratings(by='year')

        .. figure:: ../_images/robotic_vacuum_cleaners/stats/ratings_yearly.*
            :name: robotic_vacuum_cleaners_ratings_yearly
            :align: center
            :width: 80%

            Customers' ratings on robot vacuums (on a yearly basis).

        .. code-block:: python

            >>> # rvc.view_stats_on_ratings(by='month', save_as=".svg", verbose=True)
            >>> rvc.view_stats_on_ratings(by='month')

        .. figure:: ../_images/robotic_vacuum_cleaners/stats/ratings_monthly.*
            :name: robotic_vacuum_cleaners_ratings_monthly
            :align: center
            :width: 80%

            Customers' ratings on robot vacuums (on a monthly basis).

        .. code-block:: python

            >>> from src.processor import TraditionalVacuumCleaners
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> # tvc.view_stats_on_ratings(by='year', save_as=".svg", verbose=True)
            >>> tvc.view_stats_on_ratings(by='year')

        .. figure:: ../_images/traditional_vacuum_cleaners/stats/ratings_yearly.*
            :name: traditional_vacuum_cleaners_ratings_yearly
            :align: center
            :width: 80%

            Customers' ratings on traditional vacuums (on a yearly basis).

        .. code-block:: python

            >>> # tvc.view_stats_on_ratings(by='month', save_as=".svg", verbose=True)
            >>> tvc.view_stats_on_ratings(by='month')

        .. figure:: ../_images/traditional_vacuum_cleaners/stats/ratings_monthly.*
            :name: traditional_vacuum_cleaners_ratings_monthly
            :align: center
            :width: 80%

            Customers' ratings on traditional vacuums (on a monthly basis).

        .. code-block:: python

            >>> from src.processor import SmartThermostats
            >>> smt = SmartThermostats(load_prep_data=True, load_preprocd_data=False)
            >>> # smt.view_stats_on_ratings(by='year', save_as=".svg", verbose=True)
            >>> smt.view_stats_on_ratings(by='year')

        .. figure:: ../_images/smart_thermostats/stats/ratings_yearly.*
            :name: smart_thermostats_ratings_yearly
            :align: center
            :width: 80%

            Customers' ratings on smart thermostats (on a yearly basis).

        .. code-block:: python

            >>> # smt.view_stats_on_ratings(by='month', save_as=".svg", verbose=True)
            >>> smt.view_stats_on_ratings(by='month')

        .. figure:: ../_images/smart_thermostats/stats/ratings_monthly.*
            :name: smart_thermostats_ratings_monthly
            :align: center
            :width: 80%

            Customers' ratings on smart thermostats (on a monthly basis).
        """

        mpl_preferences(font_size=12, backend='TkAgg')

        plt = _check_dependency(name='matplotlib.pyplot')

        data_ = self.prep_data.copy() if data is None else data.copy()

        dat = self.get_descriptive_stats(data=data_, by=by)

        review_col_name = 'Review'

        rating_column_names = ['HighestRating', 'Rating=4*', 'Neutral', 'Rating=2*', 'LowestRating']
        r5, r4, r3, r2, r1 = [dat[col].values for col in rating_column_names]

        legend_labels = ['Highest rating', 'Rating=4*', 'Neutral', 'Rating=2*', 'Lowest rating']

        colour_5 = '#A45D5D'  # Highest rating = 5
        colour_4 = '#CEAB93'  # rating = 4
        colour_3 = '#EFEFEF'  # rating = 3
        colour_2 = '#EBD671'  # rating = 2
        colour_1 = '#FFC069'  # Lowest rating = 1
        colour_r = '#4A403A'  # Review

        fig2 = plt.figure(figsize=(5, 6) if horizontal else (7, 4), constrained_layout=True)
        ax2 = fig2.add_subplot()

        bar_width, x_ticks = 0.6, np.arange(len(dat.index))
        tick_labels = list(calendar.month_abbr)[1:] if by == 'month' else dat.index.tolist()

        if horizontal:
            b5 = ax2.barh(x_ticks, r5, color=colour_5, height=bar_width, left=r1 + r2 + r3 + r4)
            b4 = ax2.barh(x_ticks, r4, color=colour_4, height=bar_width, left=r1 + r2 + r3)
            b3 = ax2.barh(x_ticks, r3, color=colour_3, height=bar_width, left=r1 + r2)
            b2 = ax2.barh(x_ticks, r2, color=colour_2, height=bar_width, left=r1)
            b1 = ax2.barh(x_ticks, r1, color=colour_1, height=bar_width)

            handles = [b1, b2, b3, b4, b5]
            legend_labels = list(reversed(legend_labels))

            if review_stats:
                line1, = ax2.plot(
                    dat[review_col_name].values, x_ticks, color=colour_r, linestyle='--')
                handles += [line1]
                legend_labels += [review_col_name]

            ax2.set_xticks(np.linspace(0, 1, 6))  # np.arange(0., 1.1, .2)
            ax2.set_xticklabels(['{:.0%}'.format(x) for x in ax2.get_xticks()])
            ax2.set_xlim(xmin=0, xmax=1)

            ax2.set_yticks(x_ticks, tick_labels)

            ax2.set_ylabel(by.title(), fontweight='black', color='#4A403A')

        else:
            b5 = ax2.bar(x_ticks, r5, color=colour_5, width=bar_width, bottom=r1 + r2 + r3 + r4)
            b4 = ax2.bar(x_ticks, r4, color=colour_4, width=bar_width, bottom=r1 + r2 + r3)
            b3 = ax2.bar(x_ticks, r3, color=colour_3, width=bar_width, bottom=r1 + r2)
            b2 = ax2.bar(x_ticks, r2, color=colour_2, width=bar_width, bottom=r1)
            b1 = ax2.bar(x_ticks, r1, color=colour_1, width=bar_width)

            handles = [b5, b4, b3, b2, b1]

            if review_stats:
                line1, = ax2.plot(
                    x_ticks, dat[review_col_name].values, color=colour_r, linestyle='--')
                handles += [line1]
                legend_labels += ['Review']

            ax2.set_yticks(np.arange(0., 1.1, .2))  # np.arange(0., 1.1, .2)
            ax2.set_yticklabels(['{:.0%}'.format(x) for x in ax2.get_yticks()])
            ax2.set_ylim(ymin=0, ymax=1)

            ax2.set_xticks(x_ticks, tick_labels)

            ax2.set_xlabel(by.title(), fontweight='black', color='#4A403A')

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.spines['left'].set_position(('outward', 8))
        ax2.spines['bottom'].set_position(('outward', 5))

        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')

        ax2.legend(
            handles, legend_labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower center',
            borderaxespad=0, ncol=3, handletextpad=0.1, frameon=False)  # mode='expand',

        # fig2.tight_layout()

        fig2.show()

        if save_as:
            for save_as_ in {save_as, ".svg", ".pdf"}:
                fig_filename = f"ratings_{by}ly" + ("_h" if horizontal else "") + save_as_
                path_to_fig = self.cdd("stats", fig_filename, mkdir=True)
                save_data(fig2, path_to_fig, verbose=verbose, **kwargs)

                prod_name = self.PRODUCT_NAME.lower().replace(' ', '_')
                _file_path = f"{prod_name}\\stats\\{fig_filename}"
                docs_file_path = cd(f"docs\\source\\_images\\{_file_path}", mkdir=True)
                shutil.copyfile(path_to_fig, docs_file_path)

    # == Preprocess the review text ================================================================

    def _check_preprocd_data(self, refresh=False, verbose=True):
        """Check whether the preprocessed data is already loaded, or needs to be updated."""
        if self.preprocd_data is None or refresh:
            if self.prep_data is None:
                self.load_prep_data(verbose=verbose)
            self.preprocd_data = self.prep_data.copy()

    def remove_unverified_reviews(self, refresh=False, verbose=False):
        """
        Remove cases where the reviews were not verified.

        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(rvc.prep_data)
            143217
            >>> rvc.remove_unverified_reviews()
            >>> len(rvc.prep_data)  # Remove unverified reviews does not change `rvc.raw_data`
            143217
            >>> len(rvc.preprocd_data)
            129109
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(tvc.prep_data)
            230479
            >>> tvc.remove_unverified_reviews()
            >>> len(tvc.prep_data)
            230479
            >>> len(tvc.preprocd_data)
            213052

        .. note::

            This method does not make any changes to ``.prep_data``.
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        column_name = 'Verified'

        if column_name in self.preprocd_data.columns:
            if verbose:
                pre_msg = "\t" if verbose == 2 else ""
                print(pre_msg + "Removing unverified reviews", end=" ... ")

            self.preprocd_data = self.preprocd_data[self.preprocd_data[column_name]]

            if verbose:
                print("Done.")

    def remove_short_reviews(self, word_count_threshold=20, refresh=False, verbose=False):
        """
        Remove cases where the reviews were too short to provide adequate or useful information.

        :param word_count_threshold: word count in a review,
            beyond which the review is not considered for further analysis. Defaults to ``20``.
        :type word_count_threshold: int
        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners
            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(rvc.prep_data)
            143217
            >>> # Keep the cases where the word count of the review were greater than 20
            >>> rvc.remove_short_reviews(verbose=True)
            Removing short reviews (No. of words < 20) ... Done.
            >>> len(rvc.preprocd_data)
            104340
            >>> rvc.remove_short_reviews(word_count_threshold=25, verbose=True)
            Removing short reviews (No. of words < 25) ... Done.
            >>> len(rvc.preprocd_data)
            96254
            >>> rvc.remove_short_reviews(word_count_threshold=50, verbose=True)
            Removing short reviews (No. of words < 50) ... Done.
            >>> len(rvc.preprocd_data)
            64834
            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(tvc.prep_data)
            230479
            >>> # Keep the cases where the word count of the review were greater than 20
            >>> tvc.remove_short_reviews(verbose=True)
            Removing short reviews (No. of words < 20) ... Done.
            >>> len(tvc.preprocd_data)
            148003
            >>> tvc.remove_short_reviews(word_count_threshold=25, verbose=True)
            Removing short reviews (No. of words < 25) ... Done.
            >>> len(tvc.preprocd_data)
            131977
            >>> tvc.remove_short_reviews(word_count_threshold=50, verbose=True)
            Removing short reviews (No. of words < 50) ... Done.
            >>> len(tvc.preprocd_data)
            77577
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        self.word_count_threshold = word_count_threshold

        column_name = 'ReviewText'

        if column_name in self.prep_data.columns:
            if verbose:
                p = "\t" if verbose == 2 else ""
                prt_msg = p + f"Removing short reviews (No. of words < {word_count_threshold})"
                print(prt_msg, end=" ... ")

            temp = self.preprocd_data[column_name].map(normalise_text)
            mask = temp.map(lambda x: True if len(x.split()) >= word_count_threshold else False)

            if mask.sum() != 0:
                self.preprocd_data = self.preprocd_data[mask]

            if verbose:
                print("Done.")

    def remove_non_english_reviews(self, word_count_threshold=20, refresh=False, verbose=False):
        """
        Remove cases where the reviews were NOT written in English.

        :param word_count_threshold: word count in a review,
            beyond which the review is not considered for further analysis. Defaults to ``20``.
        :type word_count_threshold: int
        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(rvc.prep_data)
            143217
            >>> rvc.preprocd_data is None
            True
            >>> rvc.remove_non_english_reviews(verbose=True)
            Removing short reviews (No. of words < 20) ... Done.
            Removing non-English reviews ... Done.
            >>> len(rvc.preprocd_data)
            101608
            >>> rvc.remove_non_english_reviews(word_count_threshold=25, verbose=True)
            Removing short reviews (No. of words < 25) ... Done.
            Removing non-English reviews ... Done.
            >>> len(rvc.preprocd_data)
            93902
            >>> rvc.remove_non_english_reviews(word_count_threshold=50, verbose=True)
            Removing short reviews (No. of words < 50) ... Done.
            Removing non-English reviews ... Done.
            >>> len(rvc.preprocd_data)
            63584

            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(tvc.prep_data)
            230479
            >>> tvc.preprocd_data is None
            True
            >>> tvc.remove_non_english_reviews(verbose=True)
            Removing short reviews (No. of words < 20) ... Done.
            Removing non-English reviews ... Done.
            >>> len(tvc.preprocd_data)
            146656
            >>> tvc.remove_non_english_reviews(word_count_threshold=25, verbose=True)
            Removing short reviews (No. of words < 25) ... Done.
            Removing non-English reviews ... Done.
            >>> len(tvc.preprocd_data)
            130923
            >>> tvc.remove_non_english_reviews(word_count_threshold=50, verbose=True)
            Removing short reviews (No. of words < 50) ... Done.
            Removing non-English reviews ... Done.
            >>> len(tvc.preprocd_data)
            77196
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        # path_to_lang_detect_rslt = self.cdd(f"lang_detect_wordcount{word_count_threshold}.pickle")
        #
        # if os.path.exists(path_to_lang_detect_rslt) and not update:
        #     languages = load_pickle(path_to_lang_detect_rslt)
        # else:
        self.remove_short_reviews(word_count_threshold=word_count_threshold, verbose=verbose)

        column_name = 'ReviewText'

        if column_name in self.prep_data.columns:
            if verbose:
                print(("\t" if verbose == 2 else "") + "Removing non-English reviews", end=" ... ")

            languages = self.preprocd_data[column_name].map(identify_language)
            # save_pickle(languages, path_to_pickle=path_to_lang_detect_rslt)

            self.preprocd_data = self.preprocd_data[languages == 'English']

            del languages
            gc.collect()

            if verbose:
                print("Done.")

    def get_vader_sentiment_score(self, review_column_name=None, processes=None, refresh=False,
                                  verbose=False):
        """
        Add calculated VADER sentiment score to the preprocessed data.

        :param review_column_name: name of the column that contains preprocessed review text,
            when ``review_column_name=None``, it defaults to
            :attr:`~src.processor.ProductReviews.ORIGINAL_REVIEW_COLUMN_NAME`
        :type review_column_name: str
        :param processes: number of worker processes to use by `multiprocessing.Pool()`_,
            defaults to ``None``
        :type processes: int
        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        .. _`multiprocessing.Pool()`:
            https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> rvc.preprocd_data is None
            True
            >>> rvc.get_vader_sentiment_score(review_column_name='ReviewText', verbose=True)
            Calculating VADER sentiment scores ... Done.
            >>> len(rvc.preprocd_data)
            143217
            >>> score_cols = ['vs_neg_score', 'vs_neu_score', 'vs_pos_score', 'vs_compound_score']
            >>> rvc.preprocd_data[score_cols].head()
               vs_neg_score  vs_neu_score  vs_pos_score  vs_compound_score
            0         0.174         0.723         0.103            -0.4359
            1         0.000         0.630         0.370             0.9781
            2         0.000         0.624         0.376             0.8878
            3         0.019         0.699         0.283             0.9591
            4         0.093         0.711         0.196             0.6862

            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> tvc.preprocd_data is None
            True
            >>> tvc.get_vader_sentiment_score(review_column_name='ReviewText', verbose=True)
            Calculating VADER sentiment scores ... Done.
            >>> len(tvc.preprocd_data)
            230479
            >>> score_cols = ['vs_neg_score', 'vs_neu_score', 'vs_pos_score', 'vs_compound_score']
            >>> tvc.preprocd_data[score_cols].head()
               vs_neg_score  vs_neu_score  vs_pos_score  vs_compound_score
            0         0.063         0.790         0.147             0.7389
            1         0.093         0.713         0.194             0.9002
            2         0.156         0.686         0.158            -0.1923
            3         0.000         0.664         0.336             0.9647
            4         0.045         0.823         0.132             0.8805
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        if review_column_name is None:
            review_col_name = self.ORIGINAL_REVIEW_COLUMN_NAME
        else:
            review_col_name = review_column_name

        if review_col_name not in self.preprocd_data.columns:
            raise KeyError(
                f"The column named '{review_col_name}' is not ready in the preprocessed data.")

        if self.VADER_COLUMN_NAME not in self.preprocd_data.columns or refresh:
            if verbose:
                prt_msg = ("\t" if verbose == 2 else "") + "Calculating VADER sentiment scores"
                print(prt_msg, end=" ... ")

            # vader_score = get_vader_sentiment_score(self.preprocd_data[review_col_name])
            processes_ = os.cpu_count() - 1 if processes is None else processes
            with multiprocessing.Pool(processes=processes_) as p:
                sub_lists = np.array_split(self.preprocd_data[review_col_name], processes_)
                sub_lists_ = p.map(get_vader_sentiment_score, sub_lists)
                # vader_score = pd.concat(sub_lists_)
                vader_score = pd.concat(sub_lists_).reindex(
                    self.preprocd_data[review_col_name].index)

            vader_score_col_names = vader_score.columns.to_list()
            if any(x in vader_score_col_names for x in self.preprocd_data.columns):
                self.preprocd_data.drop(vader_score_col_names, axis=1, inplace=True)

            self.preprocd_data = pd.concat([self.preprocd_data, vader_score], axis=1)

            if verbose:
                print("Done.")

    def _sentiment_on_rating(self, refresh=False, verbose=False, **kwargs):
        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        col_name = 'rating'
        sentiment_column_name = self.SENTIMENT_COLUMN_NAME + f'_on_{col_name}'

        if sentiment_column_name not in self.preprocd_data.columns:
            if verbose:
                prt_msg = ("\t" if verbose == 2 else "") + "Determining sentiment on rating"
                print(prt_msg, end=" ... ")

            sentiment_on_rating_ = self.preprocd_data[col_name].map(
                determine_sentiment_on_rating, **kwargs)
            sentiment_on_rating = sentiment_on_rating_.to_frame(sentiment_column_name)

            self.preprocd_data = pd.concat([self.preprocd_data, sentiment_on_rating], axis=1)

            if verbose:
                print("Done.")

    def _sentiment_on_vs_score(self, review_column_name=None, refresh=False, verbose=False,
                               **kwargs):
        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        col_name = 'vs_score'
        sentiment_column_name = self.SENTIMENT_COLUMN_NAME + f'_on_{col_name}'

        if sentiment_column_name not in self.preprocd_data.columns:
            try:  # Calculate VADER sentiment score
                self.get_vader_sentiment_score(
                    review_column_name=review_column_name, refresh=refresh, verbose=verbose)
            except Exception as e:
                print("Failed. {}".format(e))

            if verbose:
                p = "\t" if verbose == 2 else ""
                print(p + "Determining sentiment on VADER sentiment score", end=" ... ")
            # Determine sentiment
            sentiment_on_vader_score_ = self.preprocd_data[self.VADER_COLUMN_NAME].map(
                determine_sentiment_on_vs_score, **kwargs)

            sentiment_on_vader_score = sentiment_on_vader_score_.to_frame(sentiment_column_name)

            self.preprocd_data = pd.concat([self.preprocd_data, sentiment_on_vader_score], axis=1)

            if verbose:
                print("Done.")

    def determine_sentiment(self, dual_scale=False, review_column_name=None, refresh=False,
                            verbose=False):
        """
        Determine the sentiment of each product review.

        :param dual_scale: Whether to consider both rating and VADER sentiment score to determine
            sentiment. Defaults to ``False``.
        :type dual_scale: bool
        :param review_column_name: name of the column that contains review text,
            when ``review_column_name=None``, it defaults to
            :attr:`~src.processor.ProductReviews.REVIEW_COLUMN_NAME`
        :type review_column_name: str
        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

        *Reviews on the robot vacuum cleaners*:

            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)

            >>> rvc.determine_sentiment(review_column_name='ReviewText', verbose=True)
            Determining sentiment on rating ... Done.
            Calculating VADER sentiment scores ... Done.
            Determining sentiment on VADER sentiment score ... Done.

            >>> rvc.preprocd_data[['sentiment_on_rating', 'sentiment_on_vs_score']].head()
              sentiment_on_rating sentiment_on_vs_score
            0            negative              negative
            1            positive              positive
            2            positive              positive
            3            positive              positive
            4            positive              positive

            >>> rvc.preprocd_data.shape
            (143217, 18)

            >>> rvc.preprocd_data_ is None
            True

            >>> rvc.determine_sentiment(dual_scale=True, review_column_name='ReviewText',
            ...                         verbose=True)
            Sentiment on rating is available.
            Sentiment on VADER sentiment score is available.
            Determining sentiment on both rating and VADER sentiment score ... Done.
            >>> rvc.preprocd_data_.shape
            (107707, 17)
            >>> (rvc.preprocd_data_.sentiment_on_dual_scale == 'positive').sum()
            91215
            >>> (rvc.preprocd_data_.sentiment_on_dual_scale == 'negative').sum()
            15652
            >>> (rvc.preprocd_data_.sentiment_on_dual_scale == 'neutral').sum()
            840

        *Reviews on the traditional vacuum cleaners*::

            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)

            >>> tvc.determine_sentiment(review_column_name='ReviewText', verbose=True)
            Determining sentiment on rating ... Done.
            Calculating VADER sentiment scores ... Done.
            Determining sentiment on VADER sentiment score ... Done.

            >>> tvc.preprocd_data[['sentiment_on_rating', 'sentiment_on_vs_score']].head()
              sentiment_on_rating sentiment_on_vs_score
            0            positive              positive
            1            positive              positive
            2            negative              negative
            3            positive              positive
            4            positive              positive

            >>> tvc.preprocd_data.shape
            (230479, 18)

            >>> tvc.preprocd_data_ is None
            True

            >>> tvc.determine_sentiment(dual_scale=True, review_column_name='ReviewText',
            ...                         verbose=True)
            Sentiment on rating is available.
            Sentiment on VADER sentiment score is available.
            Determining sentiment on both rating and VADER sentiment score ... Done.
            >>> tvc.preprocd_data_.shape
            (175153, 17)
            >>> (tvc.preprocd_data_.sentiment_on_dual_scale == 'positive').sum()
            144158
            >>> (tvc.preprocd_data_.sentiment_on_dual_scale == 'negative').sum()
            29422
            >>> (tvc.preprocd_data_.sentiment_on_dual_scale == 'neutral').sum()
            1573
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        sr_col, svs_col = map(
            lambda x: self.SENTIMENT_COLUMN_NAME + x, ['_on_rating', '_on_vs_score'])

        if sr_col not in self.preprocd_data.columns:
            try:  # Based on rating
                self._sentiment_on_rating(verbose=verbose)
            except Exception as e:
                print("Failed. {}".format(e))
        else:
            if verbose:
                print("Sentiment on rating is available.")

        if svs_col not in self.preprocd_data.columns:
            try:  # Based on VADER sentiment score
                self._sentiment_on_vs_score(review_column_name=review_column_name, verbose=verbose)
            except Exception as e:
                print("Failed. {}".format(e))
        else:
            if verbose:
                print("Sentiment on VADER sentiment score is available.")

        self.dual_scale = dual_scale

        if self.dual_scale:  # Based on both rating and VADER sentiment score
            sdm_col = self.SENTIMENT_COLUMN_NAME + f'_on_dual_scale'

            if sdm_col not in self.preprocd_data.columns:
                if verbose:
                    msg = ("\t" if verbose == 2 else "") + \
                          "Determining sentiment on both rating and VADER sentiment score"
                    print(msg, end=" ... ")

                try:
                    rating_sentiment = self.preprocd_data[sr_col]  # Rating
                    vader_sentiment = self.preprocd_data[svs_col]  # VADER sentiment score

                    mask = rating_sentiment.eq(vader_sentiment)
                    sentiment_on_dual_scale = vader_sentiment.to_frame(sdm_col)[mask]

                    temp = self.preprocd_data[mask].drop(columns=[sr_col, svs_col])
                    self.preprocd_data_ = pd.concat([temp, sentiment_on_dual_scale], axis=1)
                    self.preprocd_data_.index = range(mask.sum())

                    if verbose:
                        print("Done.")

                except Exception as e:
                    print("Failed. {}".format(e))

    @classmethod
    def _preprocess_review_text(cls, review_text, func, verbose, msg, processes=None, **kwargs):
        if verbose:
            print(("\t\t" if verbose == 2 else "\t") + msg, end=" ... ")

        processes_ = os.cpu_count() - 1 if processes is None else processes
        with multiprocessing.Pool(processes=processes_) as p:
            rev_txt = p.map(functools.partial(func, **kwargs), review_text)
            review_text_ = pd.Series(rev_txt, index=review_text.index)

        # # Alternative method:
        # review_text_ = review_text.map(lambda x: func(x, **kwargs))

        if verbose:
            print("Done.")

        return review_text_

    def preprocess_review_text(self, rm_punctuation=True, rm_stopwords=True, rm_single_letters=True,
                               rm_digits=True, lemmatize_words=True, refresh=False, verbose=False):
        """
        Process review text.

        :param rm_punctuation: Whether to remove punctuation. Defaults to ``True``.
        :type rm_punctuation: bool
        :param rm_stopwords: Whether to remove stopwords. Defaults to ``True``.
        :type rm_stopwords: bool
        :param rm_single_letters: Whether to remove single letters. Defaults to ``True``.
        :type rm_single_letters: bool
        :param rm_digits: Whether to remove digits. Defaults to ``True``.
        :type rm_digits: bool
        :param lemmatize_words: Whether to lemmatize the words in review texts;
            defaults to ``True``.
        :type lemmatize_words: bool
        :param refresh: Whether to perform this function on the raw or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> rvc.preprocess_review_text(verbose=True)
            Processing review text ...
                Removing stopwords ... Done.
                Removing punctuation ... Done.
                Removing single letters ... Done.
                Removing digits ... Done.
                Lemmatizing texts ... Done.
            Done.
            >>> len(rvc.preprocd_data)
            143217

            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> tvc.preprocess_review_text(verbose=True)
            Processing review text ...
                Removing stopwords ... Done.
                Removing punctuation ... Done.
                Removing single letters ... Done.
                Removing digits ... Done.
                Lemmatizing texts ... Done.
            Done.
            >>> len(tvc.preprocd_data)
            230479
        """

        self._check_preprocd_data(refresh=refresh, verbose=verbose)

        if self.PROCESSED_REVIEW_COLUMN_NAME not in self.preprocd_data.columns:
            if verbose:
                print(("\t" if verbose == 2 else "") + "Processing review text ... ")

            review_text = self.preprocd_data[self.ORIGINAL_REVIEW_COLUMN_NAME].str.lower()

            if rm_stopwords:
                review_text = self._preprocess_review_text(
                    review_text, remove_stopwords, verbose=verbose, msg="Removing stopwords")

            if rm_punctuation:  # Remove punctuation
                review_text = self._preprocess_review_text(
                    review_text, remove_punctuation, verbose=verbose, msg="Removing punctuation")

            if rm_single_letters:
                review_text = self._preprocess_review_text(
                    review_text, remove_single_letters, verbose=verbose,
                    msg="Removing single letters")

            if rm_digits:
                review_text = self._preprocess_review_text(
                    review_text, remove_digits, verbose=verbose, msg="Removing digits")

            if lemmatize_words:
                review_text = self._preprocess_review_text(
                    review_text, lemmatize_text, verbose=verbose, msg="Lemmatizing texts")

            review_text_ = review_text.to_frame(name=self.PROCESSED_REVIEW_COLUMN_NAME)
            review_text_.index = self.preprocd_data.index
            self.preprocd_data = pd.concat(objs=[self.preprocd_data, review_text_], axis=1)

            if int(verbose) == 1:
                print("Done.")

    def preprocess_prep_data(self, verified_reviews_only=False, word_count_threshold=20,
                             dual_scale=False, refresh=False, verbose=False, **kwargs):
        """
        Preprocess the preparatory data.

        :param verified_reviews_only: consider only the verified reviews. Defaults to ``False``.
        :type verified_reviews_only: bool
        :param word_count_threshold: word count in a review,
            beyond which the review is not considered for further analysis. Defaults to ``20``
        :type word_count_threshold: int
        :param dual_scale: indicate whether the sentiment is determined on both rating and
            VADER sentiment score. Defaults to ``False``.
        :type dual_scale: bool
        :param refresh: Whether to perform this function on the raw data or preparatory data,
            and update the preprocessed data. Defaults to ``False``.
        :type refresh: bool
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :py:meth:`~src.processor.ProductReviews.preprocess_review_text`

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

            >>> rvc = RoboticVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(rvc.prep_data)
            143217
            >>> rvc.preprocd_data is None
            True
            >>> rvc.preprocess_prep_data(verbose=True)
            Preprocessing the preparatory data ...
                Removing short reviews (No. of words < 20) ... Done.
                Removing non-English reviews ... Done.
                Determining sentiment on rating ... Done.
                Calculating VADER sentiment scores ... Done.
                Determining sentiment on VADER sentiment score ... Done.
                Processing review text ...
                    Removing stopwords ... Done.
                    Removing punctuation ... Done.
                    Removing single letters ... Done.
                    Removing digits ... Done.
                    Lemmatizing texts ... Done.
            Done.
            >>> len(rvc.preprocd_data)
            101608

            >>> tvc = TraditionalVacuumCleaners(load_prep_data=True, load_preprocd_data=False)
            >>> len(tvc.prep_data)
            230479
            >>> tvc.preprocd_data is None
            True
            >>> tvc.preprocess_prep_data(verbose=True)
            Preprocessing the preparatory data ...
                Removing short reviews (No. of words < 20) ... Done.
                Removing non-English reviews ... Done.
                Determining sentiment on rating ... Done.
                Calculating VADER sentiment scores ... Done.
                Determining sentiment on VADER sentiment score ... Done.
                Processing review text ...
                    Removing stopwords ... Done.
                    Removing punctuation ... Done.
                    Removing single letters ... Done.
                    Removing digits ... Done.
                    Lemmatizing texts ... Done.
            Done.
            >>> len(tvc.preprocd_data)
            146656
        """

        try:
            verbose_ = 2 if verbose else False

            if self.prep_data is None:
                if refresh == 2:
                    self.load_prep_data(update=True, verbose=verbose)
                else:
                    self.load_prep_data(verbose=verbose)

            if verbose:
                print("Preprocessing the preparatory data ... ")

            self.verified_reviews_only = verified_reviews_only
            self.word_count_threshold = word_count_threshold
            self.dual_scale = dual_scale

            if self.verified_reviews_only:
                self.remove_unverified_reviews(refresh=refresh, verbose=verbose_)

            self.remove_non_english_reviews(
                word_count_threshold=self.word_count_threshold, refresh=refresh, verbose=verbose_)

            self.determine_sentiment(dual_scale=self.dual_scale, verbose=verbose_)

            self.preprocess_review_text(verbose=verbose_, **kwargs)

            if verbose:
                print("Done.")

        except Exception as e:
            print("Failed to preprocess the data. {}".format(e))

    def _exchange_preprocd_data_given_dual_scale(self):
        temp = self.preprocd_data.copy()
        self.preprocd_data = self.preprocd_data_.copy()
        self.preprocd_data_ = temp

    def load_preprocd_data(self, verified_reviews_only=False, word_count_threshold=20,
                           dual_scale=False, before_date=None, update=False, verbose=False,
                           ret_data=False):
        """
        Read the preprocessed product reviews.

        :param verified_reviews_only: consider only the verified reviews. Defaults to ``False``.
        :type verified_reviews_only: bool
        :param word_count_threshold: word count in a review,
            beyond which the review is not considered for further analysis. Defaults to ``20``.
        :type word_count_threshold: int
        :param dual_scale: indicate whether the sentiment is determined on both rating and
            VADER sentiment score. Defaults to ``False``.
        :type dual_scale: bool
        :param before_date: date before which the preparatory data is considered,
            e.g. ``'2021-04-01'`` and ``'2022-04-01'``. Defaults to ``None``.
        :type before_date: str | None
        :param update: Whether to reprocess the data. Defaults to ``False``.
        :type update: bool | int
        :param verbose: Whether to print relevant information in console. Defaults to ``False``.
        :type verbose: bool | int
        :param ret_data: Whether to return the preprocessed data. Defaults to ``False``.
        :type ret_data: bool
        :return: preprocessed data of the product reviews
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.processor import RoboticVacuumCleaners, TraditionalVacuumCleaners

            >>> rvc = RoboticVacuumCleaners(load_preprocd_data=False)
            >>> # rvc.load_preprocd_data(update=True, verbose=True)  # Update preprocd_data
            >>> rvc.load_preprocd_data(verified_reviews_only=False, verbose=True)
            >>> rvc.preprocd_data.shape
            (101608, 19)
            >>> rvc.preprocd_data_ is None
            True
            >>> rvc.load_preprocd_data(verified_reviews_only=True, verbose=True)
            >>> rvc.preprocd_data.shape
            (89989, 19)
            >>> rvc.preprocd_data_ is None
            True
            >>> rvc.load_preprocd_data(dual_scale=True, verbose=True)
            >>> rvc.preprocd_data.shape  # dual_scale=True
            (77775, 18)
            >>> rvc.preprocd_data_.shape  # dual_scale=False
            (101608, 19)

            >>> tvc = TraditionalVacuumCleaners(load_preprocd_data=False)
            >>> # tvc.load_preprocd_data(update=True, verbose=True)  # Update preprocd_data
            >>> tvc.load_preprocd_data(verified_reviews_only=False, verbose=True)
            >>> tvc.preprocd_data.shape
            (146656, 19)
            >>> tvc.preprocd_data_ is None
            True
            >>> tvc.load_preprocd_data(verified_reviews_only=True, verbose=True)
            >>> tvc.preprocd_data.shape
            (131998, 19)
            >>> tvc.preprocd_data_ is None
            True
            >>> tvc.load_preprocd_data(dual_scale=True, verbose=True)
            >>> tvc.preprocd_data.shape  # dual_scale=True
            (110978, 18)
            >>> tvc.preprocd_data_.shape  # dual_scale=False
            (146656, 19)
        """

        if verified_reviews_only != self.verified_reviews_only:
            self.preprocd_data = None
            self.verified_reviews_only = verified_reviews_only

        if word_count_threshold != self.word_count_threshold:
            self.preprocd_data = None
            self.word_count_threshold = word_count_threshold

        if dual_scale != self.dual_scale:
            self.preprocd_data = None
            self.dual_scale = dual_scale

        if self.preprocd_data is None or update:
            if self.use_db:
                table_name = self.TABLE_NAME + '_preprocd'

                if self.db_instance is None:
                    self.db_instance = CustomerReviewsAnalysis()

                if self.db_instance.table_exists(table_name, self.SCHEMA_NAME) and not update:
                    if verbose:
                        verified_note = self.if_is_verified_note()
                        prt_msg = "Loading the preprocessed data of product reviews" + verified_note
                        print(prt_msg, end=" ... ")

                    sql_query = self.specify_sql_query(
                        table_name=table_name, before_date=before_date,
                        verified_reviews_only=self.verified_reviews_only)

                    self.preprocd_data = self.db_instance.read_sql_query(
                        sql_query=sql_query, parse_dates=['review_date'], keep_default_na=False,
                        true_values=['t', 'true'], false_values=['f', 'false']).sort_values(
                        self.index_names, ignore_index=True)

                    if verbose:
                        print("Done.")

                else:
                    self.preprocess_prep_data(
                        verified_reviews_only=verified_reviews_only,
                        word_count_threshold=word_count_threshold, dual_scale=dual_scale,
                        refresh=update, verbose=verbose)
                    assert isinstance(self.preprocd_data, pd.DataFrame)

                    preprocd_data = self.preprocd_data.set_index(self.index_names).sort_index()

                    self.db_instance.import_data(
                        data=preprocd_data, table_name=table_name, schema_name=self.SCHEMA_NAME,
                        index=True, if_exists='replace', method=self.db_instance.psql_insert_copy,
                        confirmation_required=False, verbose=2 if verbose else verbose)

                    self.db_instance.null_text_to_empty_string(table_name, schema_name=self.SCHEMA_NAME)

                    self.preprocd_data = preprocd_data.reset_index().sort_values(self.index_names)

            else:
                path_to_pkl = self.cdd("preprocd_data", "preprocd_data.pkl")

                if not os.path.isfile(path_to_pkl) or update:
                    self.preprocess_prep_data(
                        verified_reviews_only=verified_reviews_only,
                        word_count_threshold=word_count_threshold, dual_scale=dual_scale,
                        refresh=update, verbose=verbose)
                    assert isinstance(self.preprocd_data, pd.DataFrame)
                    save_data(self.preprocd_data, path_to_pkl, verbose=verbose)

                else:
                    self.preprocd_data = load_data(path_to_pkl, verbose=verbose)

            default_word_count_threshold, default_dual_scale = map(
                lambda x: inspect.signature(self.preprocess_prep_data).parameters[x].default,
                ['word_count_threshold', 'dual_scale'])

            if self.word_count_threshold != default_word_count_threshold:
                self.remove_non_english_reviews(word_count_threshold=self.word_count_threshold)

            if self.dual_scale != default_dual_scale:
                self.determine_sentiment(dual_scale=self.dual_scale)
                self._exchange_preprocd_data_given_dual_scale()

        else:
            if verbose:
                verified_note = self.if_is_verified_note()
                print(f"The preprocessed data{verified_note} is already loaded.")

        self._filter_data(
            name='preprocd_data', before_date=before_date,
            verified_reviews_only=self.verified_reviews_only)

        if ret_data:
            return self.preprocd_data
