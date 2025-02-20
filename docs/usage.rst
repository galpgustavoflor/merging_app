Usage Guide
==========

Step-by-Step Process
------------------

1. Loading Files
~~~~~~~~~~~~~~~

Upload your source and target files through the UI:

* Supported formats: CSV, Excel
* Automatic data type detection
* Preview of data and statistics

2. Defining Mappings
~~~~~~~~~~~~~~~~~~

Configure how your datasets should be mapped:

.. code-block:: json

    {
        "key_source": ["id"],
        "key_target": ["reference_id"],
        "mappings": {
            "amount": {
                "destinations": ["value"],
                "function": "Direct Match"
            }
        }
    }

3. Validation Rules
~~~~~~~~~~~~~~~~~

Set up data validation rules:

.. code-block:: json

    {
        "amount": {
            "validate_nulls": true,
            "validate_range": true,
            "min_value": 0
        }
    }

4. Executing Comparison
~~~~~~~~~~~~~~~~~~~~~

* Review matching results
* Analyze validation outputs
* Download reports

Advanced Features
---------------

* Large Dataset Handling
* Custom Transformations
* Export Capabilities
