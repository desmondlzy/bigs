from nerfstudio.plugins.registry_dataparser import DataParserSpecification

def get_dataparser_config() -> DataParserSpecification:
    from masksplat.dataparser import BiGSBlenderConfig
    dataparser_config = DataParserSpecification(
        config=BiGSBlenderConfig(),
        description="BiGS Synthetic dataset parser; The data is generated using Mitsuba renderer",
    )

    return dataparser_config

dataparser_config = get_dataparser_config()